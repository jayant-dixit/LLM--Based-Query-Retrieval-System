from fastapi import FastAPI
from router import router

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(router, prefix="/api/v1")