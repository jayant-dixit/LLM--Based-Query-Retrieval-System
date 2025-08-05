from sentence_transformers import SentenceTransformer
import uuid
import re

model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def split_into_clauses(text: str) -> list[str]:
    # Split on sentence-like punctuation (adjust as needed)
    return [clause.strip() for clause in re.split(r'[.;:\n]', text) if clause.strip()]

def embed_text(pages: list[str]) -> list[dict]:
    """
    Splits each page into clauses, embeds them, and prepares data for Pinecone.
    """
    all_clauses = []
    for page in pages:
        clauses = split_into_clauses(page)
        all_clauses.extend(clauses)

    vectors = model.encode(all_clauses, convert_to_numpy=True)

    return [
        {
            "id": str(uuid.uuid4()),  # Unique ID per clause
            "values": vector.tolist(),
            "metadata": {"text": all_clauses[i]}
        }
        for i, vector in enumerate(vectors)
    ]
