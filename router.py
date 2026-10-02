from llm import generate_response

def route_query(query, conversation_context=""):
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

    response = generate_response(prompt)
    lines = response.strip().splitlines()
    route = lines[0].replace("ROUTE:", "").strip()
    retrieval_query = lines[1].replace("QUERY:", "").strip()

    return route, retrieval_query
