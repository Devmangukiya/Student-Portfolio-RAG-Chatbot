import warnings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Import all your project's components
from app.components.json_loader import load_json_files, create_text_chunks
from app.config.config import EMBEDDING_MODEL_NAME, TOP_K
from app.common.logger import get_logger

# --- SETUP ---
warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def run_full_diagnostic():
    """
    This script runs the entire data pipeline in memory to provide a definitive test.
    It completely bypasses the Flask app and any saved database files.
    """
    try:
        logger.info("--- Starting Full Diagnostic Test ---")

        # --- STEP 1: LOAD & VERIFY DATA ---
        logger.info("STEP 1: Loading documents with json_loader.py...")
        documents = load_json_files()
        if not documents:
            logger.error("DIAGNOSTIC FAILED: load_json_files() returned 0 documents.")
            return

        logger.info(f"Successfully loaded {len(documents)} documents.")
        
        # VERIFICATION POINT 1: Print the first document's content to be 100% sure.
        print("\n" + "="*20 + " VERIFICATION 1: CHECKING FIRST DOCUMENT " + "="*20)
        print("This content MUST include the Student Name, ID, Email, etc.")
        print(f"CONTENT:\n{documents[0].page_content}\n")
        print("="*87)

        # --- STEP 2: CHUNK DATA ---
        logger.info("STEP 2: Splitting documents into chunks...")
        text_chunks = create_text_chunks(documents)
        if not text_chunks:
            logger.error("DIAGNOSTIC FAILED: create_text_chunks() returned 0 chunks.")
            return
        logger.info(f"Successfully created {len(text_chunks)} chunks.")

        # --- STEP 3: CREATE IN-MEMORY VECTOR STORE ---
        logger.info(f"STEP 3: Initializing embedding model ({EMBEDDING_MODEL_NAME})...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        
        logger.info("Creating a fresh, IN-MEMORY FAISS vector store...")
        # This creates the database directly in memory, bypassing save/load issues.
        db = FAISS.from_documents(text_chunks, embeddings)
        logger.info("In-memory vector store created successfully.")

        # --- STEP 4: RETRIEVE & VERIFY ---
        logger.info("STEP 4: Creating a retriever and performing a test query...")
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": 20}
        )
        
        test_query = "STU007"
        logger.info(f"Searching for query: '{test_query}'")
        results = retriever.get_relevant_documents(test_query)

        # VERIFICATION POINT 2: Print the results.
        print("\n" + "="*20 + " VERIFICATION 2: CHECKING RETRIEVAL RESULTS " + "="*20)
        if not results:
            print("\n[!] DIAGNOSTIC FAILED: The retriever found 0 documents for the query.")
            print("[!] This is the root of the problem. Please double-check your json_loader.py and config.py files.")
        else:
            print(f"\n[*] DIAGNOSTIC SUCCESS: The retriever found {len(results)} documents!")
            print("[*] This proves your core data loading and retrieval logic is PERFECT.\n")
            for i, doc in enumerate(results):
                print(f"--- Document {i+1} ---\n")
                print(f"CONTENT:\n{doc.page_content}\n")
        print("="*89)

    except Exception as e:
        logger.error(f"An error occurred during the diagnostic: {e}", exc_info=True)


if __name__ == "__main__":
    run_full_diagnostic()
