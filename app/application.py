from flask import Flask, render_template, request, jsonify
import os

from app.common.logger import get_logger
# --- CRITICAL CHANGE ---
# Import the correct function from the correct file.
from app.components.retriever import create_rag_chain

# Initialize logger
logger = get_logger(__name__)

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load RAG Chain at Startup ---
try:
    logger.info("Application starting up... Initializing RAG chain.")
    # Ensure the Hugging Face token is available before trying to create the chain
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        logger.warning("HUGGINGFACEHUB_API_TOKEN is not set. The application might fail.")
    
    # --- CRITICAL CHANGE ---
    # Call the new, correct function name.
    rag_chain = create_rag_chain()
    logger.info("RAG chain initialized successfully. Application is ready.")
except Exception as e:
    logger.critical(f"FATAL: Failed to initialize the RAG chain on startup. Error: {e}", exc_info=True)
    rag_chain = None

# --- Define Routes ---

@app.route('/')
def home():
    """
    Renders the main chat interface page.
    """
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    """
    API endpoint to get a response from the RAG chain.
    """
    if not rag_chain:
        logger.error("Cannot process request because the RAG chain failed to initialize.")
        return jsonify({"error": "The AI model is not available. Please check the server logs."}), 500

    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        logger.warning("Received a request with no query.")
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        logger.info(f"Invoking RAG chain with query: '{user_query}'")
        response_text = rag_chain.invoke(user_query)
        logger.info("Successfully received response from the chain.")
        
        return jsonify({"response": response_text})
        
    except Exception as e:
        error_message = f"An error occurred while processing the query with the RAG chain: {e}"
        logger.error(error_message, exc_info=True)
        return jsonify({"error": "An internal error occurred while generating the response."}), 500

# --- Run the Application ---

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

