from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def embed_text(pages: list[str], doc_title: str = "Uploaded PDF") -> list[dict]:
    full_text = "\n\n".join(pages)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    docs = text_splitter.create_documents([full_text])
    
    all_chunks = [doc.page_content for doc in docs]
    
    vectors = model.encode(all_chunks, convert_to_numpy=True)

    return [
        {
            "id": str(uuid.uuid4()),
            "values": vectors[i].tolist(),
            "metadata": {
                "text": all_chunks[i]
            }
        }
        for i in range(len(all_chunks))
    ]