from sentence_transformers import SentenceTransformer
import uuid
import re

model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def split_into_clauses(text: str) -> list[str]:
    return [clause.strip() for clause in re.split(r'[.;:\n]', text) if clause.strip()]

def extract_section_title(text: str) -> str:
    lines = text.strip().split("\n")
    for line in lines[:3]:
        if re.match(r'^[A-Z][A-Z\s\-0-9]{4,}$', line.strip()):
            return line.strip()
        if re.match(r'^\d+(\.\d+)*\s+[A-Z].*', line.strip()):
            return line.strip()
    return "Unknown Section"

def embed_text(pages: list[str], doc_title: str = "Uploaded PDF") -> list[dict]:
    all_clauses = []
    metadata_list = []

    for page_num, page in enumerate(pages):
        section_title = extract_section_title(page)
        clauses = split_into_clauses(page)

        for i, clause in enumerate(clauses):
            all_clauses.append(clause)
            metadata_list.append({
                "text": clause,
                "page_number": page_num + 1,
                "section_title": section_title,
                "chunk_index": page_num,
                "sentence_index": i,
                "document_title": doc_title,
                "source": "Uploaded PDF"
            })

    vectors = model.encode(all_clauses, convert_to_numpy=True)

    return [
        {
            "id": str(uuid.uuid4()),
            "values": vectors[i].tolist(),
            "metadata": metadata_list[i]
        }
        for i in range(len(all_clauses))
    ]
