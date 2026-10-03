import os
from dotenv import load_dotenv
import time
import random
from google.genai import errors

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


def generate_gemini_response(contents, model=None):
    model = model or LLM_MODEL
    gemini_contents = format_gemini_contents(contents)

    # retry temporary api failures before giving up
    max_retries = 5

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=gemini_contents
            )
            return response.text

        except (errors.ClientError, errors.ServerError) as e:
            # only retry errors that are likely to be temporary
            if e.code not in (429, 500, 502, 503, 504):
                raise

            # raise the error if all retry attempts have been used
            if attempt == max_retries - 1:
                raise

            # exponentially increase wait time and add jitter
            # to avoid repeatedly hitting the api at the same interval
            wait_time = (2 ** attempt) + random.uniform(0, 1)

            print(
                f"Gemini temporarily unavailable ({e.code}). "
                f"Retrying in {wait_time:.1f}s..."
            )

            time.sleep(wait_time)


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