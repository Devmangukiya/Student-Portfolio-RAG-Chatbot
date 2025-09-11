import os
from dotenv import load_dotenv
load_dotenv()


HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DB_FAISS_PATH = "vectorstore/db_faiss"

DATA_PATH = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 8