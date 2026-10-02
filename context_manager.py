
SUMMARY_BATCH_SIZE = 20

# build the unsummarized conversation context sent to the llm
def build_context(messages, last_summarized_id):
    if last_summarized_id is None:
        recent_messages = messages
    else:
        recent_messages = [
            message for message in messages
            if message["id"] > last_summarized_id
        ]

    gemini_history = []

    for message in recent_messages:
        role = "model" if message["role"] == "assistant" else "user"

        gemini_history.append({
            "role": role,
            "parts": [{"text": message["content"]}]
        })

    return gemini_history


# summarize messages that are leaving the recent context window
def summarize_messages(client, messages, existing_summary=""):
    conversation_text = ""

    for message in messages:
        conversation_text += f"{message['role']}: {message['content']}\n"

    prompt = f"""
    Update the conversation summary using the messages below.

    Existing summary:
    {existing_summary}

    New messages:
    {conversation_text}

    Preserve important facts, decisions, preferences, and context that may
    be useful later in the conversation. Remove or compress details that are
    no longer important. Keep the summary concise and under 500 words.
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


# update the summary when enough unsummarized messages have accumulated
def update_summary(
    supabase,
    client,
    conversation_id,
    messages,
    summary_batch_size=SUMMARY_BATCH_SIZE
):
    # get the current summary information
    result = (
        supabase.table("conversations")
        .select("summary, last_summarized_message_id")
        .eq("id", conversation_id)
        .single()
        .execute()
    )

    existing_summary = result.data["summary"] or ""
    last_summarized_id = result.data["last_summarized_message_id"]

    # find messages that have not been summarized yet
    if last_summarized_id is None:
        unsummarized_messages = messages
    else:
        unsummarized_messages = [
            message for message in messages
            if message["id"] > last_summarized_id
        ]

    # wait until a full batch has accumulated
    if len(unsummarized_messages) < summary_batch_size:
        return existing_summary, last_summarized_id

    # summarize the completed batch
    messages_to_summarize = unsummarized_messages[:summary_batch_size]

    new_summary = summarize_messages(
        client,
        messages_to_summarize,
        existing_summary
    )

    # remember the last message included in the summary
    last_summarized_id = messages_to_summarize[-1]["id"]

    supabase.table("conversations").update({
        "summary": new_summary,
        "last_summarized_message_id": last_summarized_id
    }).eq("id", conversation_id).execute()

    return new_summary, last_summarized_id