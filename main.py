import os
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client
from context_manager import build_context, update_summary
from retrieval import retrieve_chunks, build_document_context
from router import route_query
from llm import generate_response, LLMError
import time
from observability import log_request, estimate_cost


# load environment variables
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# create supabase client
supabase = create_client(supabase_url, supabase_key)

# initialize authentication state
if "user" not in st.session_state:
    st.session_state.user = None

if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None

# show login page if user is not authenticated
if st.session_state.user is None:
    st.title("Internal AI Assistant")
    st.write("Sign in to continue.")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Log in"):
        try:
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})

            st.session_state.user = response.user
            st.session_state.access_token = response.session.access_token
            st.session_state.refresh_token = response.session.refresh_token
            st.rerun()

        except Exception:
            st.error("Invalid email or password")

    st.stop()

# restore authentication after streamlit reruns
supabase.auth.set_session(st.session_state.access_token, st.session_state.refresh_token)

# get the authenticated user's role
result = (supabase.table("users").select("role").eq("id", st.session_state.user.id).single().execute())
user_role = result.data["role"]

# create streamlit webpage
st.title("Internal AI Assistant")

# get this user's previous conversations
result = (
    supabase.table("conversations")
    .select("id, title, created_at")
    .eq("user_id", st.session_state.user.id)
    .order("created_at", desc=True)
    .execute()
)

conversations = result.data

# left align sidebar button text
st.markdown("""
<style>
    [data-testid="stSidebar"] button {
        text-align: left;
        justify-content: flex-start;
    }
</style>
""", unsafe_allow_html=True)

# display previous conversations in the sidebar
st.sidebar.title("Chats")

# start a new conversation
if st.sidebar.button("+ New Chat"):
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.rerun()

for conversation in conversations:
    title = conversation["title"] or f"Conversation {conversation['id']}"

    if st.sidebar.button(title, key=f"conversation_{conversation['id']}"):
        # select this conversation
        st.session_state.conversation_id = conversation["id"]

        # load this conversation's messages from the database
        result = (
            supabase.table("messages")
            .select("id, role, content")
            .eq("conversation_id", conversation["id"])
            .order("created_at")
            .execute()
        )

        st.session_state.messages = result.data

        # rerun so the selected conversation is displayed
        st.rerun()

# initialize current conversation
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display previous messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# get a new message from the user
user_message = st.chat_input("Ask me anything")

if user_message:
    # initialize observability state in case run fails - can still track some metrics
    request_start_time = time.perf_counter()
    route = None
    retrieval_query = None
    routing_latency_ms = None
    routing_metadata = {
        "stage": None,
        "provider": None,
        "model": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "generation_latency_ms": None,
        "retry_count": 0,
        "errors_handled": []
    }
    retrieved_chunks = []
    retrieval_latency_ms = None
    llm_metadata = {
        "stage": None,
        "provider": None,
        "model": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "generation_latency_ms": None,
        "retry_count": 0,
        "errors_handled": []
    }
    final_status = "success"
    final_error = None

    try:
        # create a conversation when the first message is sent
        if st.session_state.conversation_id is None:
            result = supabase.table("conversations").insert({"user_id": st.session_state.user.id}).execute()
            st.session_state.conversation_id = result.data[0]["id"]

        # save the user's message to the database
        result = supabase.table("messages").insert({
            "conversation_id": st.session_state.conversation_id,
            "role": "user",
            "content": user_message
        }).execute()
        user_message_id = result.data[0]["id"]

        # save and display the user's message
        st.session_state.messages.append({"id": user_message_id, "role": "user", "content": user_message})
        st.chat_message("user").write(user_message)

        # update the summary if a full batch has accumulated
        summary, last_summarized_id = update_summary(supabase, st.session_state.conversation_id, st.session_state.messages)

        # build context from messages that have not been summarized
        llm_history = build_context(st.session_state.messages, last_summarized_id)

        # add the older conversation summary to the context
        if summary:
            llm_history.insert(0, {"role": "user", "content": f"Earlier conversation summary:\n{summary}"})

        # give the router recent conversation context
        conversation_context = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in st.session_state.messages[-7:-1] # exclude most recent message
        )

        # decide whether company documents are needed
        route, retrieval_query, routing_latency_ms, routing_metadata = route_query(user_message, conversation_context)

        # retrieve company information only when needed
        if route == "COMPANY_CONTEXT_REQUIRED":
            retrieved_chunks, retrieval_latency_ms = retrieve_chunks(retrieval_query, user_role)
            document_context = build_document_context(retrieved_chunks)

            llm_history.insert(0, {
                "role": "user",
                "content": f"""
                Relevant company information:
    
                {document_context}
    
                Use this information when it is relevant to the user's question.
                Cite factual claims from company documents using the provided source,
                for example [EmployeeHandbook.md].
                """
            })

        # send the managed context to the llm
        response, llm_metadata = generate_response(llm_history, stage="generation", return_metadata=True)

        # save the assistant's message to the database
        result = supabase.table("messages").insert({
            "conversation_id": st.session_state.conversation_id,
            "role": "assistant",
            "content": response
        }).execute()
        assistant_message_id = result.data[0]["id"]


    except LLMError as e:
        if e.metadata["stage"] == "routing":
            routing_metadata = e.metadata
            routing_latency_ms = e.metadata["generation_latency_ms"]
        else:
            llm_metadata = e.metadata
        final_status = "error"
        final_error = str(e)
        st.error("The AI service is temporarily unavailable.")

    except Exception as e:
        final_status = "error"
        final_error = str(e)
        st.error("Something went wrong while processing your request.")


    finally:
        # build retrieval metadata from whatever was successfully retrieved
        retrieved_documents = list(dict.fromkeys(chunk["filename"] for chunk in retrieved_chunks))
        retrieved_chunk_data = [
            {
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "similarity": chunk["similarity"]
            }
            for chunk in retrieved_chunks
        ]

        # aggregate llm usage across routing and answer generation
        input_tokens = (routing_metadata["input_tokens"] or 0) + (llm_metadata["input_tokens"] or 0)
        output_tokens = (routing_metadata["output_tokens"] or 0) + (llm_metadata["output_tokens"] or 0)
        total_tokens = (routing_metadata["total_tokens"] or 0) + (llm_metadata["total_tokens"] or 0)
        retry_count = routing_metadata["retry_count"] + llm_metadata["retry_count"]
        errors_handled = routing_metadata["errors_handled"] + llm_metadata["errors_handled"]

        total_latency_ms = int((time.perf_counter() - request_start_time) * 1000)
        provider = llm_metadata["provider"] or routing_metadata["provider"]
        model = llm_metadata["model"] or routing_metadata["model"]
        estimated_cost_usd = estimate_cost(model, input_tokens, output_tokens)

        request_data = {
            "user_id": st.session_state.user.id,
            "conversation_id": st.session_state.conversation_id,
            "route": route,
            "retrieval_query": retrieval_query,
            "retrieved_documents": retrieved_documents,
            "retrieved_chunks": retrieved_chunk_data,
            "retrieval_count": len(retrieved_chunks),
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "routing_latency_ms": routing_latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": llm_metadata["generation_latency_ms"],
            "total_latency_ms": total_latency_ms,
            "retry_count": retry_count,
            "errors_handled": errors_handled,
            "final_status": final_status,
            "final_error": final_error
        }

        log_request(request_data)

    if final_status == "success":
        # generate a title after the first user-assistant exchange
        if len(st.session_state.messages) == 1:
            title_response = generate_response(
                f"""
                Create a short title for this conversation.
                Use at most 5 words.
                Return only the title.

                User:
                {user_message}

                Assistant:
                {response}
                """,
                stage = "title"
            )

            conversation_title = title_response.strip()
            supabase.table("conversations").update({
                "title": conversation_title
            }).eq("id", st.session_state.conversation_id).execute()

        # save and display the assistant's message
        st.session_state.messages.append({"id": assistant_message_id, "role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

        # refresh the sidebar after the first exchange
        if len(st.session_state.messages) == 2:
            st.rerun()