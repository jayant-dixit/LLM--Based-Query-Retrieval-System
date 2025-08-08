import os
from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool
from utils.documentLoader import extract_text_by_page, download_pdf_from_url
from utils.embedding import embed_text
from utils.generateAnswer import generate_answer, initialize_bm25_corpus
from utils.vectorDB import upsert_embeddings, reset_pinecone_index # IMPORT THE NEW FUNCTION
import time
from fastapi import Request
from config import PINECONE_API_KEY

router = APIRouter()

class RunRequest(BaseModel):
    documents: str
    questions: list[str]

secret_token = os.getenv("SECRET_TOKEN")

@router.post("/hackrx/run")
async def run_hackrx(request: RunRequest, http_request: Request):
    
    # Get token from Authorization header
    token = None
    auth_header = http_request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split("Bearer ")[1]
    
    print("Token received:", token)
    if not token or token != secret_token:
        return {"error": "Unauthorized"}, 401
            
    try:
            pdf_path = await download_pdf_from_url(request.documents)
            print(f"PDF downloaded to {pdf_path}")

            print("Extracting text from PDF...")
            pages = extract_text_by_page(pdf_path)
            os.remove(pdf_path) # Clean up temporary file
            print(f"Extracted {len(pages)} pages.")

            print("Embedding text chunks...")
            embeddings = embed_text(pages)
            print(f"Generated {len(embeddings)} embeddings.")

            print("Resetting Pinecone index...")
            reset_pinecone_index()

            print("Upserting embeddings to Pinecone...")
            upsert_embeddings(embeddings)
            print("Embeddings upserted.")

            # --- Initialize BM25 Corpus ---
            # This is crucial for the hybrid search
            initialize_bm25_corpus(embeddings) # Pass the list of chunk dicts


            print("\nGenerating answers for queries...")
            answers = await generate_answer(request.questions)

            # for i, (query, answer) in enumerate(zip(request.questions, answers)):
            #     print(f"\n--- Query {i+1} ---")
            #     print(f"Q: {query}")
            #     print(f"A: {answer}")

            return {"answers": answers}
        
    except Exception as e:
            print(f"An error occurred: {e}")
            return {"error": str(e)}
