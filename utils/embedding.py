from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(pages: list[str]) -> list[dict]:
    """
    Embeds each page separately and prepares data for Pinecone.
    """
    vectors = model.encode(pages, convert_to_numpy=True)

    return [
        {
            "id": f"hackrx-{i}",
            "values": vector.tolist(),
            "metadata": {"text": pages[i]}
        }
        for i, vector in enumerate(vectors)
    ]

