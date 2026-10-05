import os
from dotenv import load_dotenv
import threading
import time

EMBEDDING_MIN_INTERVAL = 1.0

embedding_rate_lock = threading.Lock()
last_embedding_time = 0.0

load_dotenv()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

# initialize the configured embedding provider
if EMBEDDING_PROVIDER == "gemini":
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
else:
    raise ValueError(f"unsupported embedding provider: {EMBEDDING_PROVIDER}")


def wait_for_embedding_rate_limit():
    global last_embedding_time

    with embedding_rate_lock:
        elapsed = time.monotonic() - last_embedding_time
        wait_time = max(0, EMBEDDING_MIN_INTERVAL - elapsed)

        if wait_time > 0:
            time.sleep(wait_time)

        last_embedding_time = time.monotonic()


def generate_embedding(text, rate_limit=False):
    if rate_limit:
        wait_for_embedding_rate_limit()

    return generate_gemini_embedding(text)


def generate_gemini_embedding(text):
    # generate an embedding using gemini
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS
        )
    )

    return response.embeddings[0].values
