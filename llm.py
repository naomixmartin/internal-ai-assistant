import os
from dotenv import load_dotenv
import time
import random
from google.genai import errors


class LLMError(Exception):
    """LLM request failure that preserves observability metadata."""

    def __init__(self, message, metadata):
        super().__init__(message)
        self.metadata = metadata


load_dotenv()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

# initialize the configured llm provider
if LLM_PROVIDER == "gemini":
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
else:
    raise ValueError(f"unsupported llm provider: {LLM_PROVIDER}")


def generate_response(contents, stage, model=None, return_metadata=False):
    # route generation requests to the configured provider
    if LLM_PROVIDER == "gemini":
        return generate_gemini_response(contents, stage, model, return_metadata)

    raise ValueError(f"unsupported llm provider: {LLM_PROVIDER}")


def generate_gemini_response(contents, stage, model=None, return_metadata=False):
    model = model or LLM_MODEL
    gemini_contents = format_gemini_contents(contents)

    max_retries = 5
    retry_count = 0
    errors_handled = []
    start_time = time.perf_counter()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=gemini_contents)

            generation_latency_ms = int((time.perf_counter() - start_time) * 1000)

            # collect token usage when gemini provides it
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", None)
            output_tokens = getattr(usage, "candidates_token_count", None )
            total_tokens = getattr(usage, "total_token_count", None)

            metadata = {
                "stage": stage,
                "provider": LLM_PROVIDER,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "generation_latency_ms": generation_latency_ms,
                "retry_count": retry_count,
                "errors_handled": errors_handled
            }

            if return_metadata:
                return response.text, metadata

            return response.text

        except (errors.ClientError, errors.ServerError) as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise

            if attempt == max_retries - 1:
                # record the final failed attempt before giving up
                errors_handled.append({"code": e.code, "stage": stage})
                generation_latency_ms = int((time.perf_counter() - start_time) * 1000)

                metadata = {
                    "stage": stage,
                    "provider": LLM_PROVIDER,
                    "model": model,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "generation_latency_ms": generation_latency_ms,
                    "retry_count": retry_count,
                    "errors_handled": errors_handled
                }

                # preserve llm metrics so the failed request can still be logged
                raise LLMError(str(e), metadata) from e

            retry_count += 1
            errors_handled.append({"code": e.code, "stage": stage})
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Gemini temporarily unavailable ({e.code}). Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)


def format_gemini_contents(contents):
    # leave simple prompts unchanged
    if isinstance(contents, str):
        return contents

    # convert generic messages to gemini's message format
    gemini_contents = []

    for message in contents:
        role = "model" if message["role"] == "assistant" else "user"
        gemini_contents.append({"role": role, "parts": [{"text": message["content"]}]})

    return gemini_contents