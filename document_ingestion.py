from pathlib import Path
from pypdf import PdfReader
from docx import Document
import os
from dotenv import load_dotenv
from supabase import create_client
from embedding import generate_embedding


CHUNK_SIZE = 250
CHUNK_OVERLAP = 50

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

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

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


def ingest_document(file_path, access_level = "employee"):
    file_path = Path(file_path)

    # extract and chunk the document
    text = extract_text(file_path)
    chunks = chunk_text(text)

    # create the document record
    result = admin_supabase.table("documents").insert({
        "filename": file_path.name,
        "file_type": file_path.suffix.lower(),
        "access_level": access_level
    }).execute()

    document_id = result.data[0]["id"]

    # embed and store each chunk
    for chunk_index, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)

        admin_supabase.table("document_chunks").insert({
            "document_id": document_id,
            "chunk_index": chunk_index,
            "content": chunk,
            "embedding": embedding
        }).execute()

    return document_id
