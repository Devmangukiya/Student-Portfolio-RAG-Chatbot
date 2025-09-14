import warnings
from pathlib import Path
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.common.custom_exception import CustomException
from app.common.logger import get_logger
from app.config.config import DATA_PATH, CHUNK_OVERLAP, CHUNK_SIZE

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def metadata_func(record: dict, metadata: dict) -> dict:

    for key, value in record.items():
        if key != "content":
             metadata[key] = value if value is not None else ""
    return metadata


def load_json_files():
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        file_path = project_root / DATA_PATH / "students_achievements.json"
        logger.info(f"Attempting to load JSON file from: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Source JSON file not found at: {file_path}")

        logger.info(f"Found file '{file_path.name}'. Initializing JSONLoader...")

        jq_schema = (
            '.[] as $parent | $parent.achievements[] | '
            '{ "content": ("Student Name: " + ($parent.name // "") + ". Student ID: " + ($parent.student_id // "") + ". Email: " + ($parent.email // "") + ". Department: " + ($parent.department // "") + ". Achievement: " + (.type // "") + " - " + (.title // "") + ": " + (.description // "")), '
            '"student_name": ($parent.name // ""), "student_id": ($parent.student_id // ""), "email": ($parent.email // ""), "department": ($parent.department // ""), "achievement_id": (.achievement_id // ""), "date": (.date // ""), "status": (.status // ""), "approved_by": (.approved_by // ""), "credit_awarded": .credit_awarded }'
        )
        
    
        loader = JSONLoader(
            file_path=str(file_path),
            jq_schema=jq_schema,
            content_key="content",
            metadata_func=metadata_func, # Use the compatible function
            text_content=False
        )

        documents = loader.load()
        logger.info(f"Successfully loaded and sanitized {len(documents)} documents.")
        return documents

    except Exception as e:
        error_message = CustomException(f"Failed to load/process JSON. Error: {e}")
        logger.error(str(error_message), exc_info=True)
        raise error_message

def create_text_chunks(documents):
    if not documents:
        logger.warning("No documents to split.")
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        text_chunks = text_splitter.split_documents(documents)
        logger.info(f"Generated {len(text_chunks)} text chunks.")
        return text_chunks
    except Exception as e:
        error_message = CustomException(f"Failed to create text chunks. Error: {e}")
        logger.error(str(error_message), exc_info=True)
        raise error_message

