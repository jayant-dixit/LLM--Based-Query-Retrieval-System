from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

BATCH_SIZE = 100

def upsert_embeddings(embeddings: list):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = PINECONE_INDEX_NAME

    # Check if index exists before creating
    # CORRECTED LINE: Call the .names() method
    if index_name not in pc.list_indexes().names(): 
        # Using a fixed dimension for 'BAAI/bge-small-en-v1.5' which is 384
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )

    index = pc.Index(index_name)

    # Batch upsert to avoid exceeding Pinecone 4MB limit
    for i in range(0, len(embeddings), BATCH_SIZE):
        batch = embeddings[i:i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace="hackrx")