import os
from dotenv import load_dotenv
from supabase import create_client
from embedding import generate_embedding
from google import genai


load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


def retrieve_chunks(query, match_count=5):
    # embed the user's question
    query_embedding = generate_embedding(query)

    # find the most similar document chunks
    result = supabase.rpc(
        "match_document_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": match_count
        }
    ).execute()

    return result.data


def build_document_context(chunks):
    # format retrieved chunks with their source documents
    context = ""

    for chunk in chunks:
        context += (
            f"\n[SOURCE: {chunk['filename']}]\n"
            f"{chunk['content']}\n"
        )

    return context


def answer_with_context(query, chunks):
    # build context from the retrieved company documents
    context = build_document_context(chunks)

    prompt = f"""
    Answer the user's question using only the company information provided below.

    Cite factual claims using the source document provided with the information.
    For example: [EmployeeHandbook.md]
    
    Only cite sources that directly support the claim.

    If the company information does not contain enough information to answer
    the question, say that you could not find the answer in the company documents.

    Company information:
    {context}

    User question:
    {query}
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


if __name__ == "__main__":
    query = "How much vacation do employees get?"

    chunks = retrieve_chunks(query)
    answer = answer_with_context(query, chunks)

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")
    for chunk in chunks:
        print(
            f"{chunk['filename']} "
            f"(chunk {chunk['chunk_index']}, "
            f"similarity {chunk['similarity']:.3f})"
        )