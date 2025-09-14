import os
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

PINECONE_EMBEDDING_MODEL = "pinecone/llama-text-embed-v2"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "student-portfolio-index"

DATA_PATH = "data/"
EVAL_DATA_PATH = "eval/"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 50
TOP_K = 8