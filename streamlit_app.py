import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import all your existing, well-structured AI components
from app.common.logger import get_logger
from app.components.retriever import create_rag_chain
from app.components.data_query_engine import DataQueryEngine
from app.components.llm import load_llm
from app.config.config import GROQ_API_KEY, GROQ_MODEL_NAME

# --- Initialize Logger ---
logger = get_logger(__name__)

# --- The "Router" Prompt (same as before) ---
PROMPT_ROUTER = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are an expert query classifier. Your task is to determine the user's intent and route their question to the correct tool. You have three tools available:
1. `portfolio_summary`: For questions about a SINGLE, SPECIFIC student.
2. `general_query`: For questions that involve searching or filtering across MULTIPLE students.
3. `general_conversation`: For all other questions, greetings, or general knowledge.
Respond with ONLY the name of the correct tool.
"""),
        ("human", "{question}")
    ]
)

# --- The General Conversation Prompt (same as before) ---
PROMPT_GENERAL = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Answer the user's question concisely and directly."),
        ("human", "{question}")
    ]
)

# --- Caching the AI Components ---
# This is the most important step for performance. The @st.cache_resource decorator
# tells Streamlit to load these complex objects only once, not on every user interaction.
@st.cache_resource
def load_ai_components():
    """Loads all necessary AI components and returns them as a dictionary."""
    logger.info("Initializing all AI components for Streamlit app...")
    try:
        rag_chain = create_rag_chain()
        data_query_engine = DataQueryEngine()
        llm = load_llm(model_name=GROQ_MODEL_NAME, groq_api_key=GROQ_API_KEY)
        router_chain = PROMPT_ROUTER | llm | StrOutputParser()
        general_chain = PROMPT_GENERAL | llm | StrOutputParser()
        logger.info("All components initialized successfully.")
        return {
            "rag_chain": rag_chain,
            "data_query_engine": data_query_engine,
            "router_chain": router_chain,
            "general_chain": general_chain
        }
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize AI components on startup. Error: {e}", exc_info=True)
        return None

# --- Main Application Logic ---

# Set the page configuration (title, icon, etc.)
st.set_page_config(
    page_title="Athena | Student Portfolio AI",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🎓 Athena | Student Portfolio AI")
st.caption("Your intelligent assistant for querying student achievement records.")

# Load the AI components using the cached function
ai_components = load_ai_components()

# Initialize chat history in the session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am Athena. How can I assist you with the student records today?"}]

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from the chat box at the bottom of the screen
if prompt := st.chat_input("Ask Athena a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- The Router Logic ---
    if ai_components:
        with st.chat_message("assistant"):
            with st.spinner("Athena is thinking..."):
                try:
                    logger.info(f"Routing query: '{prompt}'")
                    route = ai_components["router_chain"].invoke({"question": prompt}).strip()
                    logger.info(f"Determined route: '{route}'")
                    
                    response = ""
                    if "portfolio_summary" in route:
                        logger.info("Invoking RAG chain for summary...")
                        result_dict = ai_components["rag_chain"].invoke(prompt)
                        response = result_dict.get('answer', "Sorry, I couldn't generate a summary.")
                    elif "general_query" in route:
                        logger.info("Invoking Data Query Engine...")
                        response = ai_components["data_query_engine"].query_data(prompt)
                    else: # Default to general conversation
                        logger.info("Invoking General Conversation chain...")
                        response = ai_components["general_chain"].invoke({"question": prompt})
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    logger.error(f"An error occurred while processing the request: {e}", exc_info=True)
                    error_message = "Sorry, an internal error occurred. Please try again."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        st.error("The AI components failed to load. Please check the server logs.")
