from pathlib import Path
from langchain_community.document_loaders import JSONLoader

from app.common.logger import get_logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.common.custom_exception import CustomException
from app.config.config import DATA_PATH, CHUNK_OVERLAP, CHUNK_SIZE

# Initialize the logger for this module
logger = get_logger(__name__)

def metadata_func(record: dict, metadata: dict) -> dict:
    """
    A helper function to extract all necessary metadata from each JSON record 
    created by the jq schema.
    """
    metadata["student_name"] = record.get("name")
    metadata["student_id"] = record.get("student_id")
    metadata["email"] = record.get("email")
    metadata["dob"] = record.get("dob")
    metadata["department"] = record.get("department")
    
    # Achievement-level metadata, now including all fields
    metadata["achievement_id"] = record.get("achievement_id")
    metadata["type"] = record.get("type")
    metadata["title"] = record.get("title")
    metadata["description"] = record.get("description")
    metadata["date"] = record.get("date")
    metadata["status"] = record.get("status")
    metadata["approved_by"] = record.get("approved_by")
    metadata["credit_awarded"] = record.get("credit_awarded")
    return metadata

def load_json_files():
    """
    Loads student achievement data, creating a separate, highly enriched LangChain 
    Document for each achievement to ensure accurate retrieval.
    """
    try:
        # --- 1. ROBUST PATH CONSTRUCTION ---
        # Creates a reliable, absolute path to your data file to prevent errors.
        project_root = Path(__file__).resolve().parent.parent.parent
        file_path = project_root / DATA_PATH / "students_achievements.json"
        logger.info(f"Attempting to load JSON file from absolute path: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"Source JSON file not found at: {file_path}")

        logger.info(f"Found file '{file_path.name}'. Initializing JSONLoader...")

        # --- 2. THE DEFINITIVE JQ SCHEMA (Corrected) ---
        # This version explicitly uses the `$parent` variable to access the achievements array.
        # This removes any ambiguity and resolves the "Cannot index array" error.
        jq_schema = (
            '.[] as $parent | $parent.achievements[] | '
            '{ "content": ("Student Name: " + $parent.name + ". Student ID: " + $parent.student_id + ". Email: " + $parent.email + ". Date of Birth: " + $parent.dob + ". Department: " + $parent.department + ". Achievement: " + .type + " - " + .title + ": " + .description + ". Status: " + .status + ". Approved By: " + .approved_by + ". Credits Awarded: " + (.credit_awarded | tostring)), '
            '"name": $parent.name, "student_id": $parent.student_id, "email": $parent.email, "dob": $parent.dob, "department": $parent.department, '
            '"achievement_id": .achievement_id, "date": .date, "status": .status, "approved_by": .approved_by, "credit_awarded": .credit_awarded }'
        )

        # --- 3. LOADER CONFIGURATION ---
        loader = JSONLoader(
            file_path=str(file_path),
            jq_schema=jq_schema,
            content_key="content",
            metadata_func=metadata_func,
            text_content=False
        )

        documents = loader.load()
        if not documents:
            logger.warning(f"File '{file_path.name}' loaded 0 documents. Check the jq_schema and file content.")
            return []

        logger.info(f"Successfully loaded {len(documents)} documents from '{file_path.name}'.")
        return documents

    except Exception as e:
        error_message = CustomException(f"Failed to load and process JSON file. Error: {e}")
        logger.error(str(error_message), exc_info=True)
        raise error_message

def create_text_chunks(documents):
    """
    Splits a list of LangChain Documents into smaller chunks.
    """
    if not documents:
        logger.warning("No documents to split. Skipping.")
        return []
    
    try:
        logger.info(f"Splitting {len(documents)} documents into chunks...")
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

