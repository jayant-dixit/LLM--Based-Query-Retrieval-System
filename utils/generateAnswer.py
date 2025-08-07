# MultipleFiles/generateAnswer.py
import asyncio
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone
from groq import Groq
from config import GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
from rank_bm25 import BM25Okapi # For sparse retrieval
import re # For tokenizing BM25 corpus

groq_client = Groq(api_key=GROQ_API_KEY)
embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Global variable to store BM25 corpus (for simplicity in this example)
bm25_corpus_tokens = [] # Store tokenized chunks for BM25
bm25_chunk_map = {} # Map BM25 index to original chunk dict
bm25_model = None

def tokenize_text_for_bm25(text: str) -> list[str]:
    """Simple tokenizer for BM25: lowercase and split by non-alphanumeric."""
    return re.findall(r'\b\w+\b', text.lower())

def initialize_bm25_corpus(chunks: list[dict]):
    """
    Initializes the BM25 corpus from the chunks.
    This should be called once after documents are embedded and before queries.
    """
    global bm25_corpus_tokens, bm25_model, bm25_chunk_map
    bm25_corpus_tokens = []
    bm25_chunk_map = {}
    for i, chunk in enumerate(chunks):
        tokenized_text = tokenize_text_for_bm25(chunk['metadata']['text'])
        bm25_corpus_tokens.append(tokenized_text)
        bm25_chunk_map[i] = chunk # Store the original chunk dict
    
    bm25_model = BM25Okapi(bm25_corpus_tokens)
    print(f"BM25 corpus initialized with {len(bm25_corpus_tokens)} documents.")


# NEW FUNCTION: Generates a hypothetical answer for a query
async def generate_hypothetical_answer(query: str) -> str:
    prompt = f"Please write a concise, hypothetical answer to the following question about a policy document. The purpose of this is to improve document retrieval, not for the answer itself. Focus on key terms and concepts.\n\nQuestion: {query}\n\nAnswer:"
    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100, # Further reduced max_tokens for very focused hypothetical answer
            temperature=0.2 # Even lower temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating hypothetical answer: {e}")
        return query # Fallback to original query if LLM call fails

def reciprocal_rank_fusion(results_lists: list[list], k=60):
    """
    Applies Reciprocal Rank Fusion (RRF) to a list of ranked lists of document IDs.
    Each result in results_lists is expected to be a list of dictionaries
    with an 'id' key.
    """
    fused_scores = {}
    for results in results_lists:
        for rank, item in enumerate(results):
            doc_id = item['id']
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (rank + k)
    
    # Sort by fused score in descending order
    reranked_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return reranked_ids

async def fetch_answer(query, hypothetical_embedding):
    # --- 1. Dense Retrieval (Pinecone) ---
    pinecone_results = []
    try:
        pinecone_query_response = index.query(
            vector=hypothetical_embedding,
            top_k=30, # Increased top_k for more candidates for RRF
            namespace="hackrx",
            include_metadata=True
        )
        pinecone_results = pinecone_query_response['matches']
    except Exception as e:
        print(f"Error querying Pinecone: {e}")

    # --- 2. Sparse Retrieval (BM25) ---
    bm25_results = []
    if bm25_model and bm25_corpus_tokens:
        tokenized_query = tokenize_text_for_bm25(query)
        bm25_scores = bm25_model.get_scores(tokenized_query)
        
        # Get top N BM25 results and map back to original chunk dicts
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:30] # Top 30 for BM25
        
        for i in top_bm25_indices:
            if i in bm25_chunk_map:
                bm25_results.append(bm25_chunk_map[i])
    
    # --- 3. Reciprocal Rank Fusion (RRF) ---
    # Prepare results for RRF: list of lists of dicts with 'id'
    pinecone_rrf_input = [{'id': m['id']} for m in pinecone_results]
    bm25_rrf_input = [{'id': m['id']} for m in bm25_results]

    fused_ids = reciprocal_rank_fusion([pinecone_rrf_input, bm25_rrf_input])

    # Create a mapping from ID to full result object for easy lookup
    all_retrieved_items = {m['id']: m for m in pinecone_results}
    for bm25_res in bm25_results: # Add BM25 results to the lookup
        all_retrieved_items[bm25_res['id']] = bm25_res

    # Get the actual items based on fused_ids, taking a larger pool for re-ranking
    fused_retrieved_items = [all_retrieved_items[doc_id] for doc_id in fused_ids if doc_id in all_retrieved_items][:20] # Top 20 for re-ranking
    
    # --- 4. Re-ranking step ---
    passages_for_reranking = [m['metadata']['text'] for m in fused_retrieved_items]
    
    if not passages_for_reranking:
        return "I could not find any relevant information in the document to answer your question."

    scores = reranker_model.predict([(query, passage) for passage in passages_for_reranking])

    # Combine results and scores, then sort by score
    reranked_results = sorted(zip(passages_for_reranking, scores), key=lambda x: x[1], reverse=True)

    # Select the top 3-5 reranked chunks for the context (experiment with 3 vs 5)
    context_chunks = [m[0] for m in reranked_results[:4]] # Using top 4 for slightly more context
    context = "\n\n".join(context_chunks)

    # --- 5. LLM Generation ---
    prompt = f"""
        You are a highly skilled and accurate Q&A system specializing in policy documents. Your task is to answer the user's question using only the facts explicitly stated in the provided context.

        Instructions:
        1. Read the provided context carefully and thoroughly.
        2. Answer the question concisely and directly, citing specific policy points or sections if possible (e.g., "As per Section 3.1...").
        3. If the answer is not explicitly available in the context, state "I don't have enough information in the provided policy document to answer this question." Do not make up any information or infer beyond what is stated.
        4. The context might contain multiple relevant sections. Synthesize them into a single, coherent answer if necessary, maintaining the formal tone of a policy document.
        5. Do not include any conversational filler or introductory phrases like "Based on the document..." or "The document states...". Just provide the direct answer.

        Context:
        {context}

        Question:
        {query}

        Answer:"""

    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300, # Slightly increased max_tokens for potentially longer policy answers
            temperature=0.1 # Even lower temperature for maximum factual adherence
        )
        answer = response.choices[0].message.content.strip()
        
        # Post-processing check for "I don't know" to ensure it's not a hallucination
        if "I don't have enough information" in answer and len(context.strip()) < 50: # If context is very small
            pass # Allow "I don't know" if context was minimal
        elif "I don't have enough information" in answer and len(context.strip()) > 50:
            # If LLM says it doesn't know but there was substantial context,
            # it might be a failure to extract. For a hackathon, you might
            # want to force it to try harder or log this for debugging.
            # For now, we'll let it pass, but this is a critical area for improvement.
            print("Warning: LLM stated no info despite substantial context.")

        print(f"Q: {query}\nRetrieved Context:\n{context}\nA: {answer}")
        return answer
    except Exception as e:
        print(f"Error generating answer from LLM: {e}")
        return "An error occurred while generating the answer."

async def generate_answer(queries: list[str]) -> list[str]:
    # First, generate hypothetical answers for all queries concurrently
    hypothetical_answers = await asyncio.gather(*[generate_hypothetical_answer(q) for q in queries])
    
    # Then, embed the hypothetical answers
    query_embeddings = embedding_model.encode(hypothetical_answers, convert_to_numpy=True).tolist()

    # run all LLM calls concurrently
    tasks = [fetch_answer(query, embedding) for query, embedding in zip(queries, query_embeddings)]
    return await asyncio.gather(*tasks)

