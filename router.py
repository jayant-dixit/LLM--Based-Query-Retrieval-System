import os
from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool
from utils.documentLoader import extract_text_by_page, download_pdf_from_url
from utils.embedding import embed_text
from utils.generateAnswer import generate_answer
from utils.vectorDB import upsert_embeddings
import time
from fastapi import Request

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
            
    start = time.time()
    print("⏳ Start")

    document_path = await download_pdf_from_url(request.documents)
    print("✅ PDF downloaded in", time.time() - start)

    pages = await run_in_threadpool(extract_text_by_page, document_path)
    print("✅ Text extracted in", time.time() - start)

    embeddings = await run_in_threadpool(embed_text, pages)
    print("✅ Embeddings created in", time.time() - start)

    await run_in_threadpool(upsert_embeddings, embeddings)
    print("✅ Embeddings upserted in", time.time() - start)

    answers = await generate_answer(request.questions)
    print("✅ Answers generated in", time.time() - start)

    return {"answers": answers}
