import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
EMBEDDING_PROVIDER = "gemini"
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_embedding(text):
    # route embedding requests to the configured provider
    if EMBEDDING_PROVIDER == "gemini":
        return generate_gemini_embedding(text)

    raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")


def generate_gemini_embedding(text):
    # generate an embedding using gemini
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS
        )
    )

    return response.embeddings[0].values

if __name__ == "__main__":
    embedding = generate_embedding("Employees receive paid vacation each year.")

    print(f"embedding dimensions: {len(embedding)}")
    print(f"first 10 values: {embedding[:10]}")