import warnings
import pandas as pd
from pathlib import Path
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig
from app.components.json_loader import load_json_files, create_text_chunks
from app.components.llm import load_llm
from app.config.config import GROQ_API_KEY, GROQ_MODEL_NAME, EVAL_DATA_PATH, EMBEDDING_MODEL_NAME
from app.common.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def generate_evaluation_dataset():
    
    try:
        logger.info("--- Starting Test Set Generation (with Rate Limit control) ---")

        documents = load_json_files()
        text_chunks = create_text_chunks(documents)
        if not text_chunks:
            logger.error("No text chunks found. Aborting.")
            return

        generator_llm = load_llm(model_name=GROQ_MODEL_NAME, groq_api_key=GROQ_API_KEY)
        critic_llm = load_llm(model_name=GROQ_MODEL_NAME, groq_api_key=GROQ_API_KEY)
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )

        generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            embeddings
        )

        run_config = RunConfig(max_workers=1)

        logger.info("Generating a smaller test set to stay within API token limits...")
       
        testset = generator.generate_with_langchain_docs(
            documents=text_chunks, 
            test_size=3,  
            distributions={simple: 1.0}, 
            run_config=run_config,
            is_async=False
        )
        logger.info("Test set generation complete.")

        df = testset.to_pandas()
        output_dir = Path(EVAL_DATA_PATH)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "synthetic_test_set.csv"

        df.to_csv(output_path, index=False)
        logger.info(f"Evaluation dataset saved successfully to '{output_path}'.")

    except Exception as e:
        logger.error(f"An error occurred during test set generation: {e}", exc_info=True)


if __name__ == "__main__":
    generate_evaluation_dataset()
