import os
import time
from dotenv import load_dotenv
from supabase import create_client
from embedding import generate_embedding
from llm import generate_response

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


def retrieve_chunks(query, user_role, match_count=5):
    start_time = time.perf_counter()

    # embed the user's question
    query_embedding = generate_embedding(query)

    # find the most similar document chunks
    result = supabase.rpc(
        "match_document_chunks",
        {
            "query_embedding": query_embedding,
            "user_role": user_role,
            "match_count": match_count
        }
    ).execute()

    retrieval_latency_ms = int((time.perf_counter() - start_time) * 1000)

    return result.data, retrieval_latency_ms


def build_document_context(chunks):
    # assign one source number per unique document
    source_map = {}
    filename_to_source = {}
    context = ""

    for chunk in chunks:
        filename = chunk["filename"]

        if filename not in filename_to_source:
            source_number = len(source_map) + 1
            filename_to_source[filename] = source_number
            source_map[source_number] = filename

        source_number = filename_to_source[filename]
        context += f"\n[SOURCE {source_number}]\n{chunk['content']}\n"

    return context, source_map


def answer_with_context(query, chunks):
    # build context from the retrieved company documents
    context, source_map = build_document_context(chunks)

    prompt = f"""
    Answer the user's question using only the company information provided below.

    Cite factual claims using the numbered source that supports the claim.
    For example: [1]
    
    Only cite source numbers provided in the company information.
    Do not include filenames or the word "SOURCE" inside citations.
    
    Only cite sources that directly support the claim.

    If the company information does not contain enough information to answer
    the question, say that you could not find the answer in the company documents.

    Company information:
    {context}

    User question:
    {query}
    """

    return generate_response(prompt, stage="answer")
