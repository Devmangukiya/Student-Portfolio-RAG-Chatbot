import warnings
from pathlib import Path
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from app.components.llm import load_llm
from app.config.config import GROQ_API_KEY, GROQ_MODEL_NAME, DB_FAISS_PATH, TOP_K, EMBEDDING_MODEL_NAME
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

# --- SETUP ---
warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# --- PROMPT TEMPLATES ---

# Prompt 1: For generating the final answer
PROMPT_FINAL_ANSWER = PromptTemplate(
    template="""
### ROLE ###
You are an expert AI assistant who generates student portfolio summaries based ONLY on the provided context.

### TASK ###
Analyze the context below, which contains a list of a student's achievements. Synthesize this information into a concise, well-written portfolio summary.

### IMPORTANT RULES ###
1.  **NEVER** make up information. If the context is empty, you MUST state that you cannot find any records for the requested student.
2.  Begin the summary directly. Do not add conversational fluff.
3.  Combine achievements into natural paragraphs.

### CONTEXT ###
{context}

### USER'S QUESTION ###
{question}

### YOUR SUMMARY ###
""",
    input_variables=["context", "question"],
)

# Prompt 2 (NEW): For rewriting the user's query for better retrieval
PROMPT_REWRITE_QUERY = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert at rephrasing user questions into descriptive, standalone queries for a vector database. Your queries should be rich with keywords to improve semantic search."),
        ("human", "Rephrase the following user question to be a standalone, descriptive query. Question: {question}")
    ]
)

# --- HELPER FUNCTIONS ---

def load_vector_store():
    # ... (This function is correct, no changes needed)
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        db_path = project_root / DB_FAISS_PATH
        if not db_path.exists():
            logger.error(f"Vector store not found at {db_path}. Please run ingest_data.py first.")
            return None
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'}
        )
        db = FAISS.load_local(str(db_path), embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS Vector store loaded successfully.")
        return db
    except Exception as e:
        logger.critical(f"Failed to load FAISS vector store. Error: {e}", exc_info=True)
        return None

# --- MAIN CHAIN CREATION ---

def create_rag_chain():
    """
    Creates the complete RAG chain with an added query-rewriting step for improved retrieval accuracy.
    """
    try:
        logger.info("Initializing the portfolio generation RAG chain with query rewriting...")
        llm = load_llm(model_name=GROQ_MODEL_NAME, groq_api_key=GROQ_API_KEY)
        db = load_vector_store()

        if llm is None or db is None:
            raise CustomException("Failed to load the LLM or Vector Store.")

        # This retriever is now correct. The issue was the query, not the search type.
        retriever = db.as_retriever(search_kwargs={"k": TOP_K})

        # --- THE NEW QUERY-REWRITING CHAIN ---
        # This sub-chain takes the original question, sends it to the LLM to be rewritten,
        # and then passes the new, better query to the retriever.
        rewritten_query_retriever = (
            PROMPT_REWRITE_QUERY
            | llm
            | StrOutputParser()
            | retriever
        )

        # --- THE FINAL RAG CHAIN ---
        # It now uses the rewritten query to get the context.
        rag_chain = (
            {"context": rewritten_query_retriever, "question": RunnablePassthrough()}
            | PROMPT_FINAL_ANSWER
            | llm
            | StrOutputParser()
        )
        
        logger.info("RAG chain with query rewriting created successfully.")
        return rag_chain

    except Exception as e:
        error_message = CustomException(f"Failed to create portfolio chain: {e}")
        logger.critical(str(error_message), exc_info=True)
  
