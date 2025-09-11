  Student Portfolio AI Assistant
- An intelligent chatbot designed to provide instant, accurate answers about student achievements from a JSON dataset.
- This project leverages a sophisticated hybrid AI model combining a Retrieval-Augmented Generation (RAG) pipeline for portfolio summaries and a direct data query engine for specific, database-style questions.

1) Overview :- 
  - This application serves as a powerful interface for querying complex student data through a simple, conversational chat interface.      Instead of manually searching through JSON files or writing database queries, users can ask questions in natural language and           receive immediate, context-aware responses.

  - The system is built on a "Router of Chains" architecture, which first analyzes the user's intent and then directs the query to the      appropriate tool:

  - RAG Chain ("The Librarian"): For open-ended questions asking for a portfolio or summary about a specific student. It retrieves all      relevant achievements and uses a Large Language Model (LLM) to generate a cohesive summary.


2) Key Features :- 

  - Conversational Interface: Simple and intuitive chat-based UI built with Flask and Tailwind CSS.

  - Hybrid AI Model: Intelligently routes queries to either a RAG pipeline or a direct data query engine for optimal accuracy and           performance.

  - Natural Language Understanding: Ask complex questions in plain English, such as "Give me a summary for the student with email           rlynn@hotmail.com" or "How many students are in the Civil department?"

  - Accurate Data Retrieval: Utilizes FAISS vector store and advanced retrieval strategies (MMR) to find the most relevant information.

  - Robust & Scalable: Built with a modular architecture, making it easy to extend with new data sources or capabilities.


3) Tech Stack :-

  - Backend: Python, Flask

  - AI/LLM Framework: LangChain

  - LLM: Llama 3.1 (via Groq for high-speed inference)

  Vector Store: FAISS

  - Embeddings: sentence-transformers/all-MiniLM-L6-v2

  Frontend: HTML, CSS 


4) Getting Started :-

Follow these instructions to set up and run the project locally.

  1. Prerequisites
     Python 3.10 or higher

  2. Clone the Repository
     
  3. Create a Virtual Environment
     It is highly recommended to use a virtual environment to manage dependencies.

python -m venv venv
# On Windows
    venv\Scripts\activate
# On macOS/Linux
    source venv/bin/activate

  4. Install Dependencies
      Install all the required Python libraries.

pip install -r requirements.txt

(Note: You will need to create a requirements.txt file by running pip freeze > requirements.txt in your activated environment.)

  5. Set Up Environment Variables
      You will need API keys for the services used in this project. Create a file named .env in the root of your project directory and        add the following:

# .env file
GROQ_API_KEY="your_groq_api_key_here"

  6. Data Ingestion (Crucial First Step)
     Before you can run the application, you must build the vector store from your JSON data. This is a one-time process that you only       need to re-run if your students_achievements.json file changes.

Important: Make sure your data/students_achievements.json file is present and correctly formatted.

Run the ingestion script from the root directory:

python app/components/data_loader.py

This will create a vectorstore/db_faiss directory containing your indexed data.

  7. Run the Application
      Once the data ingestion is complete, you can start the Flask web server.

python app/application.py

The application will now be running. Open your web browser and navigate to:
http://127.0.0.1:5000

Now, It's time for start asking questions to your AI assistant!
