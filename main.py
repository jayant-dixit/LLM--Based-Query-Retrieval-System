from fastapi import FastAPI
import os
from router import router
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))  # Render sets PORT env variable
    )
