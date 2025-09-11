from langchain_community.vectorstores import FAISS
import os
from app.components.embeddings import get_embedding_model

from app.common.custom_exception import CustomException
from app.common.logger import get_logger
from app.config.config import DB_FAISS_PATH
from pathlib import Path

logger = get_logger(__name__)

def load_vector_store():
    """
    Loads an existing FAISS vector store from disk using the official embedding model.
    """
    try:
        embedding_model = get_embedding_model()
        
        # Use a robust, absolute path to prevent errors
        project_root = Path(__file__).resolve().parent.parent.parent
        db_path = project_root / DB_FAISS_PATH

        if db_path.exists():
            logger.info(f"Loading existing Vectorstore from: {db_path}")
            return FAISS.load_local(
                str(db_path),
                embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            logger.warning(f"No vector store found at {db_path}. Please run the ingestion script first.")
            return None
            
    except Exception as e:
        error_message = CustomException("Failed to load vector store", e)
        logger.error(str(error_message), exc_info=True)
        return None


def save_vector_store(text_chunks):
    """
    Creates and saves a new FAISS vector store using the official embedding model.
    """
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided to create vector store.")
        
        logger.info("Creating new vector store...")
        embedding_model = get_embedding_model()
        db = FAISS.from_documents(text_chunks, embedding_model)

        project_root = Path(__file__).resolve().parent.parent.parent
        db_path = project_root / DB_FAISS_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        logger.info(f"Saving vector store to: {db_path}")
        db.save_local(str(db_path))
        logger.info("Vector store saved successfully.")
        return db

    except Exception as e:
        error_message = CustomException("Failed to save new vector store", e)
        logger.error(str(error_message), exc_info=True)
        return None