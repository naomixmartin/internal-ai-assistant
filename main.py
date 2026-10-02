import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from supabase import create_client
from context_manager import build_context, update_summary


# load keys
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# create Gemini and supabase clients
client = genai.Client(api_key=api_key)
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
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            st.session_state.user = response.user
            st.session_state.access_token = response.session.access_token
            st.session_state.refresh_token = response.session.refresh_token
            st.rerun()

        except Exception:
            st.error("Invalid email or password")

    st.stop()

# restore authentication after streamlit reruns
supabase.auth.set_session(
    st.session_state.access_token,
    st.session_state.refresh_token
)

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
    # create a conversation when the first message is sent
    if st.session_state.conversation_id is None:
        result = supabase.table("conversations").insert({
            "user_id": st.session_state.user.id
        }).execute()

        st.session_state.conversation_id = result.data[0]["id"]

    # save the user's message to the database
    result = supabase.table("messages").insert({
        "conversation_id": st.session_state.conversation_id,
        "role": "user",
        "content": user_message
    }).execute()
    user_message_id = result.data[0]["id"]

    # save and display the user's message
    st.session_state.messages.append({
        "id": user_message_id,
        "role": "user",
        "content": user_message
    })
    st.chat_message("user").write(user_message)

    # update the summary if a full batch has accumulated
    summary, last_summarized_id = update_summary(
        supabase,
        client,
        st.session_state.conversation_id,
        st.session_state.messages
    )

    # build context from messages that have not been summarized
    gemini_history = build_context(
        st.session_state.messages,
        last_summarized_id
    )

    # add the older conversation summary to the context
    if summary:
        gemini_history.insert(0, {
            "role": "user",
            "parts": [{
                "text": f"Earlier conversation summary:\n{summary}"
            }]
        })

    # send the managed context to gemini
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=gemini_history
    )

    # save the assistant's message to the database
    result = supabase.table("messages").insert({
        "conversation_id": st.session_state.conversation_id,
        "role": "assistant",
        "content": response.text
    }).execute()
    assistant_message_id = result.data[0]["id"]

    # generate a title after the first user-assistant exchange
    if len(st.session_state.messages) == 1:
        title_response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=f"""
            Create a short title for this conversation.
            Use at most 5 words.
            Return only the title.

            User:
            {user_message}

            Assistant:
            {response.text}
            """
        )

        conversation_title = title_response.text.strip()

        supabase.table("conversations").update({
            "title": conversation_title
        }).eq(
            "id", st.session_state.conversation_id
        ).execute()

    # save and display the assistant's message
    st.session_state.messages.append({
        "id": assistant_message_id,
        "role": "assistant",
        "content": response.text
    })
    st.chat_message("assistant").write(response.text)

    # refresh the sidebar after the first exchange
    if len(st.session_state.messages) == 2:
        st.rerun()