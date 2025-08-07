import asyncio
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from groq import Groq
from config import GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

groq_client = Groq(api_key=GROQ_API_KEY)
embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


# NEW FUNCTION: Generates a hypothetical answer for a query
async def generate_hypothetical_answer(query: str) -> str:
    prompt = f"Please write a comprehensive, hypothetical answer to the following question about the provided document. The purpose of this is to improve document retrieval, not for the answer itself.\n\nQuestion: {query}\n\nAnswer:"
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

async def fetch_answer(query, embedding):
    # Retrieve top 10 results to give the reranker more options
    results = index.query(
        vector=embedding, # This embedding is now for the hypothetical answer
        top_k=10,
        namespace="hackrx",
        include_metadata=True
    )

    # Re-ranking step
    passages = [m['metadata']['text'] for m in results['matches']]
    scores = reranker_model.predict([(query, passage) for passage in passages])

    # Combine results and scores, then sort by score
    reranked_results = sorted(zip(results['matches'], scores), key=lambda x: x[1], reverse=True)

    # Select the top 3 reranked chunks for the context
    context_chunks = [m[0]['metadata']['text'] for m in reranked_results[:3]]
    context = "\n\n".join(context_chunks)

    # UPDATED PROMPT
    prompt = f"""
        You are a highly skilled and accurate Q&A system. Your task is to answer the user's question using only the facts from the provided context.

        Instructions:
        1. Read the provided context carefully.
        2. Answer the question concisely and directly.
        4. Do not make up any information.
        5. The context might contain multiple relevant sections. Synthesize them into a single, coherent answer if necessary.

        Context:
        {context}

        Question:
        {query}

        Answer:"""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",  # Or another Groq-compatible model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0.7 # Increased temperature for more creative/inferential answers
    )

    print(f"Q: {query}\nRetrieved Context:\n{context}\nA: {response.choices[0].message.content.strip()}")

    return response.choices[0].message.content.strip()

async def generate_answer(queries: list[str]) -> list[str]:
    # First, generate hypothetical answers for all queries concurrently
    hypothetical_answers = await asyncio.gather(*[generate_hypothetical_answer(q) for q in queries])
    
    # Then, embed the hypothetical answers
    query_embeddings = embedding_model.encode(hypothetical_answers, convert_to_numpy=True).tolist()

    # run all LLM calls concurrently
    tasks = [fetch_answer(query, embedding) for query, embedding in zip(queries, query_embeddings)]
    return await asyncio.gather(*tasks)