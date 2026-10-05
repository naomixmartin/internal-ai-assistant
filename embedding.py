import os
from dotenv import load_dotenv

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


def generate_embedding(text):
    # route embedding requests to the configured provider
    if EMBEDDING_PROVIDER == "gemini":
        return generate_gemini_embedding(text)

    raise ValueError(f"unsupported embedding provider: {EMBEDDING_PROVIDER}")


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
