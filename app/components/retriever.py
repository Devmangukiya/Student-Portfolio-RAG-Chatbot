import warnings
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from app.components.llm import load_llm
from app.config.config import (
    GROQ_API_KEY, 
    GROQ_MODEL_NAME, 
    PINECONE_API_KEY, 
    PINECONE_INDEX_NAME, 
    TOP_K, 
    EMBEDDING_MODEL_NAME
)
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


PROMPT_FINAL_ANSWER = PromptTemplate(
    template="""
### ROLE ###
You are an expert University Registrar's assistant, tasked with answering questions about student records.
Your tone is professional, helpful, and you are an expert at finding and presenting information clearly.

### TASK ###
Based ONLY on the context provided below, answer the user's question.

### IMPORTANT RULES ###
- If the context does not contain the answer, state that the information is not available in the provided records.
- For requests about a specific student, synthesize all their achievements into a clear, professional summary written in a natural paragraph.
- For specific questions, provide a direct and complete answer in a full sentence.

### CONTEXT ###
{context}

### USER'S QUESTION ###
{question}

### PROFESSIONAL ANSWER ###
""",
    input_variables=["context", "question"],
)


def load_vector_store():
    try:
        logger.info(f"Connecting to Pinecone index: {PINECONE_INDEX_NAME}")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'}
        )
        
        db = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME, 
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )
        return db
    except Exception as e:
        logger.critical(f"Failed to connect to Pinecone vector store. Error: {e}", exc_info=True)
        return None


def rerank_documents(inputs: dict) -> dict:
    query = inputs["question"]
    documents = inputs["context"]
    potential_names = [word for word in query.split() if word.istitle() and len(word) > 2]
    if not potential_names:
        return {"context": documents[:TOP_K], "question": query}
    reranked_docs = []
    for doc in documents:
        for name in potential_names:
            if name.lower() in doc.page_content.lower():
                reranked_docs.append(doc)
                break
    logger.info(f"Re-ranked {len(documents)} initial docs down to {len(reranked_docs)} precise matches.")
    inputs['context'] = reranked_docs[:TOP_K]
    return inputs


def create_rag_chain():
    try:
        logger.info("Initializing the RAG chain with re-ranking...")
        llm = load_llm(model_name=GROQ_MODEL_NAME, groq_api_key=GROQ_API_KEY)
        db = load_vector_store()
        if llm is None or db is None:
            raise CustomException("Failed to load the LLM or Vector Store.")

        retriever = db.as_retriever(search_kwargs={"k": 20})

        answer_generator = (PROMPT_FINAL_ANSWER | llm | StrOutputParser())

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(rerank_documents)
            | RunnableParallel(
                answer=answer_generator,
                context=lambda x: x["context"],
            )
        )
        logger.info("RAG chain with re-ranking created successfully.")
        return rag_chain
    except Exception as e:
        error_message = CustomException(f"Failed to create portfolio chain: {e}")
        logger.critical(str(error_message), exc_info=True)
        raise error_message