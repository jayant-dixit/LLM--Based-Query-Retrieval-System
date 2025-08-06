import asyncio
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
from config import GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

groq = Groq(api_key=GROQ_API_KEY)
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

async def fetch_answer(query, embedding):
    results = index.query(
                    vector=embedding,
                    top_k=5,
                    namespace="hackrx",
                    filter={"source": {"$eq": "Uploaded PDF"}},
                    include_metadata=True
    )               

    
    context = "\n\n".join([
        f"[Page {m['metadata'].get('page_number')}] {m['metadata']['text']}" 
        for m in results['matches']])

    prompt = f"""
            You are a professional assistant. Use the provided context to answer the question.

            Instructions:
            - Use only the facts provided.
            - Refer to the most relevant chunk.
            - Be concise and specific.

            Context:
            {context}

            Question:
            {query}

            Answer:"""



    response = groq.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.2
    )

    print(f"Q: {query}\nRetrieved Context:\n{context}\nA: {response}")

    return response.choices[0].message.content.strip()

async def generate_answer(queries: list[str]) -> list[str]:
    query_embeddings = model.encode(queries, convert_to_numpy=True).tolist()

    # run all LLM calls concurrently
    tasks = [fetch_answer(query, embedding) for query, embedding in zip(queries, query_embeddings)]
    return await asyncio.gather(*tasks)
