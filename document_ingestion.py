from pathlib import Path
from pypdf import PdfReader
from docx import Document
import os
from dotenv import load_dotenv
from supabase import create_client
from embedding import generate_embedding
from concurrent.futures import ThreadPoolExecutor
import random
import time

CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
EMBEDDING_WORKERS = 5

embedding_executor = ThreadPoolExecutor(max_workers=EMBEDDING_WORKERS)

load_dotenv()
admin_supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


def extract_text(file_path):
    file_path = Path(file_path)
    file_type = file_path.suffix.lower()

    # extract plain text and markdown files
    if file_type in [".md", ".txt"]:
        return file_path.read_text(encoding="utf-8")

    # extract text from pdf files
    if file_type == ".pdf":
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text

    # extract text from word documents
    if file_type == ".docx":
        document = Document(file_path)

        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]

        return "\n".join(paragraphs)

    raise ValueError(f"Unsupported file type: {file_type}")


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # split the document into words
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # move forward while preserving overlap with the previous chunk
        start += chunk_size - overlap

    return chunks


def get_source_id(file_path):
    """
    Extract the dataset document ID from an EnterpriseRAG filename.
    """
    return Path(file_path).stem.split("__", 1)[0]


def get_existing_document_id(source_type, source_id):
    result = admin_supabase.table("documents").select("id").eq(
        "source_type", source_type
    ).eq(
        "source_id", source_id
    ).limit(1).execute()

    if result.data:
        return result.data[0]["id"]

    return None


def ingest_document_once(file_path, source_type, access_level="employee"):
    file_path = Path(file_path)
    source_id = get_source_id(file_path)

    # skip documents that have already been ingested
    existing_document_id = get_existing_document_id(source_type, source_id)

    if existing_document_id is not None:
        return existing_document_id

    # extract and chunk the document
    text = extract_text(file_path)
    chunks = chunk_text(text)

    # generate embeddings using the shared worker pool
    embeddings = list(embedding_executor.map(generate_embedding, chunks))

    # prepare chunks for the atomic database write
    chunk_rows = []

    for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_rows.append({"chunk_index": chunk_index, "content": chunk, "embedding": embedding})

    # insert the document and all chunks in one transaction
    result = admin_supabase.rpc(
        "insert_document_with_chunks",
        {
            "p_filename": file_path.name,
            "p_file_type": file_path.suffix.lower(),
            "p_access_level": access_level,
            "p_source_type": source_type,
            "p_source_id": source_id,
            "p_chunks": chunk_rows
        }
    ).execute()

    return result.data


def ingest_document(file_path, source_type, access_level="employee", max_retries=3):
    file_path = Path(file_path)

    for attempt in range(max_retries):
        try:
            return ingest_document_once(file_path, source_type, access_level)

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"failed to ingest {file_path.name} after {max_retries} attempts: {e}")
                raise

            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"failed to ingest {file_path.name}: {e}")
            print(f"retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)


def ingest_documents(file_paths, source_type, access_level="employee", max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        document_ids = list(executor.map(lambda file_path: ingest_document(file_path, source_type, access_level), file_paths))

    return document_ids


if __name__ == "__main__":
    confluence_dir = Path("enterprise_rag_data/confluence")
    test_files = list(confluence_dir.glob("*.txt"))

    start_time = time.perf_counter()

    print(f"found {len(test_files)} files to ingest")

    document_ids = ingest_documents(test_files, source_type="confluence")

    # for file_path, document_id in zip(test_files, document_ids):
    #     print(f"created document {document_id}: {file_path.name}")

    elapsed = time.perf_counter() - start_time
    print(f"test ingestion complete in {elapsed:.1f} seconds")