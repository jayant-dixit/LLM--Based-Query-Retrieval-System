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

async def fetch_answer(query, embedding):
    # Retrieve top 10 results to give the reranker more options
    results = index.query(
        vector=embedding,
        top_k=10,
        namespace="hackrx",
        filter={"source": {"$eq": "Uploaded PDF"}},
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
        You are a highly skilled and helpful legal and insurance agent for Bajaj HackRx. Your goal is to answer questions based on the provided documents.

        Instructions:
        1. **Prioritize the context:** First, check the provided context for the answer.
        2. **Act as an expert:** If the context is incomplete or does not contain a direct answer, use your expertise as a legal and insurance agent to infer a helpful and logical response.
        3. **Provide a confident answer:** Do not say "I don't know" or "The information is not available." Instead, provide the most plausible and helpful answer based on the context and your professional persona.
        4. **Be concise:** Keep your answers to the point.
        
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
    query_embeddings = embedding_model.encode(queries, convert_to_numpy=True).tolist()

    # run all LLM calls concurrently
    tasks = [fetch_answer(query, embedding) for query, embedding in zip(queries, query_embeddings)]
    return await asyncio.gather(*tasks)