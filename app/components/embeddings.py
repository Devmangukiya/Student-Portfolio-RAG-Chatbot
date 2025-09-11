from langchain_huggingface import HuggingFaceEmbeddings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import EMBEDDING_MODEL_NAME
logger = get_logger(__name__)

def get_embedding_model():
    try:

        logger.info("Initializing our embedding Model")
        model = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL_NAME,
                                      model_kwargs={'device': 'cpu'})
        logger.info("Huggingface Embedding Model loaded Successfully")
        return model
    
    except Exception as e:
        error_message = CustomException("Error Occured while loading Embedding Model",e)
        logger.error(error_message)
        raise error_message