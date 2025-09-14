import pandas as pd
import warnings
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings

# Import all necessary components
from app.components.retriever import create_rag_chain
from app.components.llm import load_llm
from app.config.config import EVAL_DATA_PATH, GROQ_API_KEY, GROQ_MODEL_NAME, EMBEDDING_MODEL_NAME
from app.common.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def run_evaluation():
    
    try:
        logger.info("--- Starting RAG Pipeline Evaluation ---")

        # 1. Load the evaluation dataset
        eval_path = Path(EVAL_DATA_PATH) / "synthetic_test_set.csv"
        if not eval_path.exists():
            logger.error(f"'{eval_path}' not found. Please run generate_testset.py first.")
            return
        df = pd.read_csv(eval_path)
        if 'ground_truth_answer' in df.columns:
            df = df.rename(columns={"ground_truth_answer": "ground_truth"})

        # 2. Initialize all components
        rag_chain = create_rag_chain()
        if rag_chain is None:
            logger.error("RAG chain failed to initialize. Aborting evaluation.")
            return
            
        evaluation_llm = load_llm(model_name=GROQ_MODEL_NAME, groq_api_key=GROQ_API_KEY)
        evaluation_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )

        results = []
        for index, row in df.iterrows():
            question = row['question']
            logger.info(f"Processing question {index + 1}/{len(df)}: '{question}'")
            result = rag_chain.invoke(question)
            results.append({
                "question": question,
                "ground_truth": row['ground_truth'],
                "answer": result.get('answer', ''),
                "contexts": [doc.page_content for doc in result.get('context', [])]
            })
        logger.info("Finished processing all test questions.")

        results_df = pd.DataFrame(results)
        dataset = Dataset.from_pandas(results_df)

        logger.info("Running RAGAs evaluation...")
        run_config = RunConfig(max_workers=1)
        
        result_scores = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
            llm=evaluation_llm,
            embeddings=evaluation_embeddings,
            run_config=run_config,
            raise_exceptions=False
        )
        logger.info("RAGAs evaluation complete.")

        print("\n" + "="*20 + " RAGAs Evaluation Results " + "="*20)
        print(result_scores)
        print("="*66 + "\n")
        
        results_df = result_scores.to_pandas()
        results_path = Path(EVAL_DATA_PATH) / "evaluation_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Evaluation results saved to '{results_path}'")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}", exc_info=True)


if __name__ == "__main__":
    run_evaluation()

