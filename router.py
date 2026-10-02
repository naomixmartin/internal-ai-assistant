from google import genai


def route_query(client, query, conversation_context=""):
    # decide whether company documents are needed and create a retrieval query
    prompt = f"""
    You are routing messages for an internal company AI assistant.

    Decide whether the user's message requires company-specific information.

    Use COMPANY_CONTEXT_REQUIRED when:
    - the user explicitly asks about the company, its policies, employees,
      procedures, benefits, roles, contracts, onboarding, or organization
    - company-specific information could materially improve or change the answer
    - the conversation context shows that the user is continuing a
      company-specific discussion

    Use GENERAL when company-specific information would not meaningfully
    improve the answer.

    When genuinely uncertain, prefer COMPANY_CONTEXT_REQUIRED.

    If company context is required, also rewrite the user's message as a
    self-contained search query. Use the conversation context to resolve
    references such as "it", "that", or "they".

    Do not answer the user's question.

    Return exactly this format:

    ROUTE: GENERAL
    QUERY: <original user message>

    or:

    ROUTE: COMPANY_CONTEXT_REQUIRED
    QUERY: <self-contained retrieval query>

    Conversation context:
    {conversation_context}

    User message:
    {query}
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    lines = response.text.strip().splitlines()

    route = lines[0].replace("ROUTE:", "").strip()
    retrieval_query = lines[1].replace("QUERY:", "").strip()

    return route, retrieval_query

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    test_queries = [
        "How much PTO do employees get?",
        "What is 2 + 2?",
        "Write me a poem about cats.",
        "How do performance reviews work?",
        "What is a performance review?",
        "How should I prepare for my first day?"
    ]

    for query in test_queries:
        route = route_query(client, query)
        print(f"{query} -> {route}")