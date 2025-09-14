from flask import Flask, render_template, request, jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.common.logger import get_logger
from app.components.retriever import create_rag_chain
from app.components.data_query_engine import DataQueryEngine
from app.components.llm import load_llm
from app.config.config import GROQ_API_KEY, GROQ_MODEL_NAME

logger = get_logger(__name__)
app = Flask(__name__)

PROMPT_ROUTER = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are an expert query classifier. Your task is to determine the user's intent and route their question to the correct tool.

You have three tools available:
1. `portfolio_summary`: Use this for questions asking for a summary, portfolio, or detailed information about a SINGLE, SPECIFIC student (e.g., by name, ID, or email).
2. `general_query`: Use this for any question that involves searching, filtering, or listing information across MULTIPLE students from the student records database.
3. `general_conversation`: Use this for all other questions, including greetings, chit-chat, or general knowledge questions that are NOT about the student data (e.g., "what is machine learning?", "who is the president?").

Based on the user's question, respond with ONLY the name of the correct tool.
"""),
        ("human", "{question}")
    ]
)

PROMPT_GENERAL = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Answer the user's question concisely and directly."),
        ("human", "{question}")
    ]
)


try:
    logger.info("Application starting up... Initializing all AI components.")
    rag_chain = create_rag_chain()
    data_query_engine = DataQueryEngine()
    llm = load_llm(model_name=GROQ_MODEL_NAME, groq_api_key=GROQ_API_KEY)
    
    router_chain = PROMPT_ROUTER | llm | StrOutputParser()
    general_chain = PROMPT_GENERAL | llm | StrOutputParser()
    
    logger.info("All components initialized successfully. Application is ready.")
except Exception as e:
    logger.critical(f"FATAL: Failed to initialize AI components on startup. Error: {e}", exc_info=True)
    rag_chain = data_query_engine = router_chain = general_chain = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    if not router_chain:
        return jsonify({"error": "The AI model is not available. Please check server logs."}), 500

    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        logger.info(f"Routing query: '{user_query}'")
        route = router_chain.invoke({"question": user_query}).strip()
        logger.info(f"Determined route: '{route}'")

        if "portfolio_summary" in route:
            logger.info("Invoking RAG chain for summary...")
            result_dict = rag_chain.invoke(user_query)
            response_text = result_dict.get('answer', "Sorry, I couldn't generate a summary.")
        elif "general_query" in route:
            logger.info("Invoking Data Query Engine...")
            response_text = data_query_engine.query_data(user_query)
        else: 
            logger.info("Invoking General Conversation chain...")
            response_text = general_chain.invoke({"question": user_query})

        return jsonify({"response": response_text})
        
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while generating the response."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

