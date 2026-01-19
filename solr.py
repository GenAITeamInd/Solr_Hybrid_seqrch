import os
import uuid
import sys
import pdfplumber
import docx
import pysolr
import requests  
from datetime import datetime

# ========== 1. Configure Ollama API ========== #

OLLAMA_BASE_API_URL = "http://azdtapimanager.azure-api.net/ollama"
OLLAMA_API_KEY = ""
OLLAMA_MODEL = "nomic-embed-text"  # replace with the specific Ollama model

# ========== 2. Solr Setup ========== #

solr = pysolr.Solr(
    "http://localhost:8983/solr/core5",
    always_commit=True,
    timeout=10
)

# ========== 3. Extract Text ========== #

def extract_text(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file_path.lower().endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type")

# ========== 4. Text Chunking ========== #

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 400):
    """
    Splits text into overlapping chunks.
    chunk_size & overlap are in characters (safe approximation for tokens).
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# ========== 5. Get Embedding from Ollama ========== #

def get_embedding(text: str) -> list:
    truncated_text = text[:3000]  # safety limit

    headers = {
        'api-key': OLLAMA_API_KEY,
        'Content-Type': 'application/json'
    }
    
    data = {
        "model": OLLAMA_MODEL,
        "input": truncated_text
    }

    response = requests.post(
        f"{OLLAMA_BASE_API_URL}/api/embed",
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        embedding = response.json().get('embeddings')
        return embedding
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

# ========== 6. Upload File with Chunking ========== #

def upload_document(file_path: str):
    print(f"📄 Processing: {file_path}")

    try:
        full_text = extract_text(file_path)
        if not full_text.strip():
            print("⚠️ Skipped: No extractable text.")
            return

        # ---- Metadata input ---- #
        author = input("Enter Author name (or leave blank): ").strip() or "Unknown"
        brand = input("Enter Brand (or leave blank): ").strip() or "Unknown"
        doc_type = input("Enter Document Type (or leave blank): ").strip() or "General"
        date_input = input("Enter Date of Publish (YYYY-MM-DD) (or leave blank for today): ").strip()

        if date_input:
            try:
                date_of_publish = datetime.strptime(
                    date_input, "%Y-%m-%d"
                ).strftime("%Y-%m-%dT00:00:00Z")
            except ValueError:
                print("⚠️ Invalid date format. Using today's date.")
                date_of_publish = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")
        else:
            date_of_publish = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")

        # ---- Chunking ---- #
        chunks = chunk_text(full_text)
        print(f"🔹 Created {len(chunks)} chunks")

        source_doc_id = str(uuid.uuid4())
        solr_docs = []

        for idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)

            if embedding:  # Proceed only if embedding is returned successfully
                solr_docs.append({
                    "id": f"{source_doc_id}_{idx}",
                    "source_doc_id": source_doc_id,
                    "chunk_index": idx,
                    "title": os.path.basename(file_path),
                    "content_text": chunk,
                    "content_embedding": embedding,
                    "author": author,
                    "brand": brand,
                    "type": doc_type,
                    "date_of_publish": date_of_publish
                })

        solr.add(solr_docs)
        print(f"✅ Uploaded {len(solr_docs)} chunks to Solr.")

    except Exception as e:
        print(f"❌ Failed to upload {file_path}: {e}")

# ========== 7. Entry Point ========== #

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python upload_document.py path/to/file")
        sys.exit(1)

    upload_document(sys.argv[1])
