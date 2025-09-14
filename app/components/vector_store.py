import warnings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings 

from app.common.custom_exception import CustomException
from app.common.logger import get_logger
from app.config.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_MODEL_NAME

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def get_embedding_model():
    """
    Initializes and returns a powerful HuggingFace embedding model that matches the
    1024 dimensions required by the Pinecone index.
    """
    logger.info(f"Initializing HuggingFace embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    logger.info("HuggingFace embedding model loaded successfully.")
    return embeddings


def load_vector_store():
    """
    Connects to an existing Pinecone index to be used as the vector store.
    """
    try:
        logger.info(f"Connecting to existing Pinecone index: {PINECONE_INDEX_NAME}")
        embeddings = get_embedding_model()
        
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME, 
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
        
        logger.info("Successfully connected to Pinecone vector store.")
        return vectorstore
            
    except Exception as e:
        error_message = CustomException("Failed to connect to Pinecone vector store", e)
        logger.error(str(error_message), exc_info=True)
        return None


def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided to save to vector store.")
        
        logger.info(f"Uploading documents to Pinecone index: {PINECONE_INDEX_NAME}")
        embeddings = get_embedding_model()

        db = PineconeVectorStore.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            pinecone_api_key=PINECONE_API_KEY
        )
        
        logger.info("Successfully saved documents to Pinecone.")
        return db

    except Exception as e:
        error_message = CustomException("Failed to save to Pinecone vector store", e)
        logger.error(str(error_message), exc_info=True)
        return None

