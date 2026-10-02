import os
from dotenv import load_dotenv

load_dotenv()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

# initialize the configured llm provider
if LLM_PROVIDER == "gemini":
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
else:
    raise ValueError(f"unsupported llm provider: {LLM_PROVIDER}")


def generate_response(contents, model=None):
    # route generation requests to the configured provider
    if LLM_PROVIDER == "gemini":
        return generate_gemini_response(contents, model)

    raise ValueError(f"unsupported llm provider: {LLM_PROVIDER}")


def format_gemini_contents(contents):
    # leave simple prompts unchanged
    if isinstance(contents, str):
        return contents

    # convert generic messages to gemini's message format
    gemini_contents = []

    for message in contents:
        role = "model" if message["role"] == "assistant" else "user"

        gemini_contents.append({
            "role": role,
            "parts": [{"text": message["content"]}]
        })

    return gemini_contents


def generate_gemini_response(contents, model=None):
    # generate a response using gemini
    model = model or LLM_MODEL

    gemini_contents = format_gemini_contents(contents)

    response = client.models.generate_content(
        model=model,
        contents=gemini_contents
    )

    return response.text