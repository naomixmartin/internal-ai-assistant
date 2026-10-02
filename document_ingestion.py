from pathlib import Path
from pypdf import PdfReader
from docx import Document


CHUNK_SIZE = 250
CHUNK_OVERLAP = 50


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

if __name__ == "__main__":
    file_path = "data/EmployeeHandbook.md"

    text = extract_text(file_path)
    chunks = chunk_text(text)

    print(f"document length: {len(text.split())} words")
    print(f"number of chunks: {len(chunks)}")

    print("\nfirst chunk:")
    print(chunks[0])

    print("\nlast chunk:")
    print(chunks[-1])