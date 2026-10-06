import math
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

EMBED_MODEL = "gemini-embedding-001"

notes = [
    "Karim sent 500 USD to his brother in Herat for marriage expenses",
    "Customer asked about buying Pakistani rupees for a trip to Peshawar",
    "Ahmad paid back his old debt of 20,000 AFN",
    "Urgent transfer to Kandahar for hospital bills",
    "Zia exchanged 1,000 euros to afghani",
    "Bilal sent money to Dubai to buy mobile phones for his shop",
    "Customer complained that the Mazar transfer arrived late",
    "School fees sent to Kabul for children's education",
]

def embed(texts, task):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task),
    )
    return [e.values for e in result.embeddings]

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    size_a = math.sqrt(sum(x * x for x in a))
    size_b = math.sqrt(sum(y * y for y in b))
    return dot / (size_a * size_b)

print("Preparing notes...")
note_vectors = embed(notes, "RETRIEVAL_DOCUMENT")
print(f"Ready! {len(notes)} notes loaded.")

while True:
    query = input("\nSearch: ")
    if query.lower() == "exit":
        break

    try:
        query_vector = embed([query], "RETRIEVAL_QUERY")[0]
    except Exception as e:
        print("AI error:", e)
        continue

    scores = [cosine_similarity(query_vector, v) for v in note_vectors]
    ranked = sorted(zip(scores, notes), reverse=True)

    THRESHOLD = 0.62
    good = [(score, note) for score, note in ranked if score >= THRESHOLD]

    if not good:
        print("\nNo good match found.")
        continue

    print("\nTop matches:")
    for score, note in good[:3]:
        print(f"  {score:.2f}  {note}")