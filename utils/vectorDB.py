from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME


def upsert_embeddings(embeddings: list):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name = PINECONE_INDEX_NAME

    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=len(embeddings[0][1]),  # embedding size
            metric="cosine",
            spec={"cloud": "aws", "region": "us-east-1"}
        )

    index = pc.Index(index_name)
    
    # Upsert to namespace "hackrx"
    index.upsert(vectors=embeddings, namespace="hackrx")
