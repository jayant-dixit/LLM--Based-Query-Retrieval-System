# MultipleFiles/embedding.py
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def embed_text(pages: list[str], doc_title: str = "Uploaded PDF") -> list[dict]:
    full_text = "\n\n".join(pages)
    
    # Define separators that respect document structure, common in policies
    # Prioritize larger structural breaks first
    separators = [
        "\n\n\n", # Very large breaks (e.g., between major sections)
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ". ",    # Sentence endings
        "? ",
        "! ",
        " ",     # Word breaks
        ""       # Character fallback
    ]

    # Use RecursiveCharacterTextSplitter directly with a more robust separator list.
    # For policy documents, sometimes keeping slightly larger chunks that encompass
    # a full policy point or sub-section is better than breaking at every sentence.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=950, # Slightly increased chunk size, experiment with this
        chunk_overlap=200, # Increased overlap to ensure context is not lost
        separators=separators
    )
    
    docs = text_splitter.create_documents([full_text])
    
    all_chunks = [doc.page_content for doc in docs]
    
    print(all_chunks)
    
    vectors = model.encode(all_chunks, convert_to_numpy=True)

    return [
        {
            "id": str(uuid.uuid4()),
            "values": vectors[i].tolist(),
            "metadata": {
                "text": all_chunks[i],
                "doc_title": doc_title,
                # Potentially add page number here if extract_text_by_page provided it
                # "page_number": page_number_for_chunk (requires more complex logic)
            }
        }
        for i in range(len(all_chunks))
    ]
