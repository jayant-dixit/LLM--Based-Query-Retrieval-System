# MultipleFiles/vectorDB.py
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
import time

BATCH_SIZE = 100

def reset_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = PINECONE_INDEX_NAME

    # Check if index exists and delete it
    if index_name in pc.list_indexes().names():
        print(f"Index '{index_name}' exists, deleting...")
        pc.delete_index(index_name)
        # Wait for the index to be deleted
        time.sleep(5)
        print(f"Index '{index_name}' deleted.")

    # Recreate the index
    print(f"Creating new index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension for BAAI/bge-small-en-v1.5
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
    )
    print(f"New index '{index_name}' created.")

def upsert_embeddings(embeddings: list):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = PINECONE_INDEX_NAME
    index = pc.Index(index_name)

    # Batch upsert to avoid exceeding Pinecone 4MB limit
    for i in range(0, len(embeddings), BATCH_SIZE):
        batch = embeddings[i:i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace="hackrx")
    print(f"Upserted {len(embeddings)} embeddings to Pinecone.")

