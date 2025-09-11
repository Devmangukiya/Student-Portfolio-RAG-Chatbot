import warnings
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger

# --- SETUP ---
warnings.filterwarnings("ignore")
logger = get_logger(__name__)

def run_test():
    """
    A simple, direct test of the vector store and retriever.
    """
    logger.info("--- Running Retriever Diagnostic Tool ---")
    
    # 1. Load the vector store, exactly as the main app does.
    db = load_vector_store()
    
    if db is None:
        logger.error("Diagnostic failed: The vector store could not be loaded.")
        logger.error("This confirms the database file (e.g., vectorstore/db_faiss) does not exist.")
        logger.error("Please run your ingestion script (e.g., ingest_data.py) to create it.")
        return

    # 2. Create a retriever from the loaded database.
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # 3. Define a test query.
    test_query = "STU007"
    logger.info(f"Performing a similarity search for the query: '{test_query}'")
    
    # 4. Perform the search and get the results.
    try:
        results = retriever.get_relevant_documents(test_query)
        
        # 5. Print the results. This is the most important part.
        print("\n" + "="*30 + " RETRIEVAL RESULTS " + "="*30)
        
        if not results:
            print("\n[!] The retriever found 0 documents.")
            print("[!] This is the root cause of your problem.")
            print("[!] It confirms the data in your vector store is not correctly formatted.")
            print("[!] You MUST delete the old database and re-run your ingestion script.")
        else:
            print(f"\n[*] The retriever found {len(results)} documents successfully!\n")
            for i, doc in enumerate(results):
                print(f"--- Document {i+1} ---\n")
                print(f"CONTENT:\n{doc.page_content}\n")
                print(f"METADATA:\n{doc.metadata}\n")
                print("="*79)

    except Exception as e:
        logger.error(f"An error occurred during the search: {e}", exc_info=True)

if __name__ == "__main__":
    run_test()
