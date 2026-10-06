import json
import math
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()
client = genai.Client()

MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
PDF_FILE = "book.pdf"
CACHE_FILE = "book_chunks.json"

def load_chunks(path, chunk_size=800, overlap=150):
    reader = PdfReader(path)
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        start = 0
        while start < len(text):
            piece = text[start:start + chunk_size]
            chunks.append({"page": page_number, "text": piece})
            start += chunk_size - overlap
    return chunks

def embed(texts, task):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task),
    )
    return [e.values for e in result.embeddings]

def embed_all(texts, batch_size=50):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        while True:
            try:
                vectors += embed(batch, "RETRIEVAL_DOCUMENT")
                break
            except Exception as e:
                if "429" in str(e):
                    print("Free limit reached. Waiting 60 seconds...")
                    time.sleep(60)
                else:
                    raise
        print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        time.sleep(1)
    return vectors

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    size_a = math.sqrt(sum(x * x for x in a))
    size_b = math.sqrt(sum(y * y for y in b))
    return dot / (size_a * size_b)

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        saved = json.load(f)
    chunks, vectors = saved["chunks"], saved["vectors"]
    print("Loaded saved book.")
else:
    print("Reading the PDF...")
    chunks = load_chunks(PDF_FILE)
    print(f"Created {len(chunks)} chunks. Embedding...")
    vectors = embed_all([c["text"] for c in chunks])
    with open(CACHE_FILE, "w") as f:
        json.dump({"chunks": chunks, "vectors": vectors}, f)
    print("Book saved.")

system_prompt = """You answer questions using ONLY the book excerpts provided.
Explain in simple language.
After each fact, cite the page number like this: (page 12).
If the excerpts do not contain the answer, say exactly:
"I couldn't find this in the book."
Never use outside knowledge and never guess."""

while True:
    question = input("\nQuestion: ")
    if question.lower() == "exit":
        break

    try:
        query_vector = embed([question], "RETRIEVAL_QUERY")[0]
        scores = [cosine_similarity(query_vector, v) for v in vectors]
        ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
        top = ranked[:5]

        context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for score, c in top)
        prompt = f"Book excerpts:\n{context}\n\nQuestion: {question}"

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
    except Exception as e:
        print("AI error:", e)
        continue

        if response.text is None:
            print("\n⚠ The AI returned no answer.")
        if response.prompt_feedback:
            print("Prompt feedback:", response.prompt_feedback)
        if response.candidates:
            print("Finish reason:", response.candidates[0].finish_reason)
        continue

    print("\nAnswer:", response.text)
    pages = sorted({c["page"] for score, c in top})
    print("Searched pages:", ", ".join(str(p) for p in pages))

    