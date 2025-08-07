import os
import dotenv

dotenv.load_dotenv()

# Load environment variables from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")