from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
import re
import uuid
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def split_into_clauses(text):
    # Split on punctuation marks like '.', ';', or newlines (tune as needed)
    clauses = re.split(r'[.;\n]', text)
    return [clause.strip() for clause in clauses if clause.strip()]

def prepare_embeddings(document_text):
    clauses = split_into_clauses(document_text)
    embeddings = []
    for clause in clauses:
        emb = model.encode(clause).tolist()
        embeddings.append((str(uuid.uuid4()), emb, {"text": clause}))
    return embeddings

BATCH_SIZE = 100  # adjust if needed to stay under 4 MB per request

def upsert_embeddings(embeddings: list):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = PINECONE_INDEX_NAME

    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=len(embeddings[0]["values"]),  # fixed from [1] to ["values"]
            metric="cosine",
            spec={"cloud": "aws", "region": "us-east-1"}
        )

    index = pc.Index(index_name)

    # Batch upsert to avoid exceeding Pinecone 4MB limit
    for i in range(0, len(embeddings), BATCH_SIZE):
        batch = embeddings[i:i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace="hackrx")
