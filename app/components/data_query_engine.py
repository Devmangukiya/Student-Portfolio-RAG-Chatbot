import pandas as pd
import json
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.components.llm import load_llm
from app.config.config import GROQ_API_KEY, GROQ_MODEL_NAME, DATA_PATH
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)


# --- PROMPT 1: The Internal Router ---
# This new prompt is the "brain" that decides which method to use.
PROMPT_QUERY_TYPE_ROUTER = PromptTemplate.from_template(
    """
### ROLE ###
You are an expert query analyst. Your task is to classify a user's question as either a 'simple_filter' or a 'complex_analysis'.

### DEFINITIONS ###
- `simple_filter`: A query that involves filtering text columns based on an exact match (e.g., department is 'chemistry', type is 'workshop').
- `complex_analysis`: A query that involves numerical operations (count, sum, average, > <), or any other more complex analysis that requires flexible code.

### TASK ###
Based on the user's question, respond with ONLY the classification: `simple_filter` or `complex_analysis`.

### EXAMPLES ###
User Question: "give me list of all students which have workshop type achievement" -> simple_filter
User Question: "how many students are in the chemistry department?" -> complex_analysis
User Question: "list achievements that awarded more than 3 credits" -> complex_analysis

### USER QUESTION ###
{question}

### CLASSIFICATION ###
"""
)

# --- PROMPT 2: The Reliable JSON Method for Simple Filters ---
PROMPT_STRUCTURED_QUERY = PromptTemplate.from_template(
    """
### ROLE ###
You are an expert data analyst. Your task is to parse a user's question and extract filtering criteria into a structured JSON format.

### CONTEXT ###
You are working with a dataset with columns like: ['name', 'department', 'type', 'status'].

### TASK ###
Based on the user's question, identify the column to filter on, the value to filter by, and the column to return.

### IMPORTANT RULES ###
1. Your output MUST be ONLY a valid JSON object.
2. The `filter_value` should be the specific text from the user's query, converted to lowercase.
3. The `column_to_return` is the information the user wants listed.

### USER QUESTION ###
{question}

### JSON OUTPUT ###
"""
)

# --- PROMPT 3: The Powerful Pandas Method for Complex Analysis ---
PROMPT_PANDAS_HELPER = PromptTemplate.from_template(
    """
### ROLE ###
You are an expert Python Pandas programmer. Your task is to convert a user's question into a single, executable line of Pandas code.

### CONTEXT ###
You are working with a Pandas DataFrame named `df`.
The DataFrame has columns: ['name', 'email', 'student_id', 'department', 'type', 'status', 'credit_awarded'].
The string columns 'name', 'department', 'type', 'status' are already lowercase.

### TASK ###
Generate a single line of Python code to query the `df` to answer the question.

### IMPORTANT RULES ###
1. Your output MUST be only the single line of Python code.
2. For string comparisons, use lowercase values (e.g., `df['type'] == 'workshop'`).

### USER QUESTION ###
{question}

### PANDAS CODE ###
"""
)


class DataQueryEngine:
    def __init__(self):
        try:
            logger.info("Initializing Hybrid DataQueryEngine...")
            project_root = Path(__file__).resolve().parent.parent.parent
            file_path = project_root / DATA_PATH / "students_achievements.json"
            
            raw_data = pd.read_json(file_path)
            self.df = pd.json_normalize(raw_data.to_dict('records'), 'achievements', ['name', 'email', 'student_id', 'department', 'dob'])
            
            # Pre-process the data for consistent, case-insensitive matching
            self.df['name'] = self.df['name'].str.lower()
            self.df['department'] = self.df['department'].str.lower()
            self.df['type'] = self.df['type'].str.lower()
            self.df['status'] = self.df['status'].str.lower()
            
            logger.info("Student data loaded and sanitized successfully.")
            
            llm = load_llm(model_name=GROQ_MODEL_NAME, groq_api_key=GROQ_API_KEY)
            
            # Initialize all three chains
            self.router_chain = PROMPT_QUERY_TYPE_ROUTER | llm | StrOutputParser()
            self.structured_query_chain = PROMPT_STRUCTURED_QUERY | llm | StrOutputParser()
            self.pandas_helper_chain = PROMPT_PANDAS_HELPER | llm | StrOutputParser()
            
        except Exception as e:
            raise CustomException(f"Failed to initialize DataQueryEngine: {e}")

    def query_data(self, user_query: str) -> str:
        try:
            logger.info(f"Classifying query: '{user_query}'")
            query_type = self.router_chain.invoke({"question": user_query}).strip()
            logger.info(f"Determined query type: '{query_type}'")

            if "simple_filter" in query_type:
                return self._execute_structured_query(user_query)
            else: # Default to complex analysis
                return self._execute_pandas_query(user_query)

        except Exception as e:
            logger.error(f"An unexpected error occurred in query_data router: {e}", exc_info=True)
            return "Sorry, I encountered an error while processing your query."

    def _execute_structured_query(self, user_query: str) -> str:
        json_response = ""
        try:
            logger.info("Executing structured (JSON) query path...")
            json_response = self.structured_query_chain.invoke({"question": user_query})
            query_params = json.loads(json_response)
            
            col_filter = query_params['column_to_filter']
            val_filter = query_params['filter_value']
            col_return = query_params['column_to_return']

            result_series = self.df[self.df[col_filter] == val_filter][col_return]
            result = result_series.unique().tolist()
            
            if not result: return "No results found for that query."
            
            formatted_result = [f"{i+1}. {str(item).title()}" for i, item in enumerate(result)]
            return "Here are the results:\n" + "\n".join(formatted_result)

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Structured query failed. Response: '{json_response}'. Error: {e}")
            return "Sorry, I had trouble understanding the structure of that query."

    def _execute_pandas_query(self, user_query: str) -> str:
        generated_code = ""
        try:
            logger.info("Executing complex (Pandas) query path...")
            generated_code = self.pandas_helper_chain.invoke({"question": user_query.lower()}).strip()
            
            logger.info(f"Executing generated code: {generated_code}")
            df = self.df
            result = eval(generated_code)
            
            if isinstance(result, (pd.Series, list)):
                if not result if not isinstance(result, pd.Series) else result.empty:
                    return "No results found for that query."
                formatted_result = [str(item).title() for item in result]
                return "Here are the results:\n" + "\n".join(formatted_result)
            elif isinstance(result, pd.DataFrame):
                return result.to_markdown(index=False)
            else:
                return str(result)

        except Exception as e:
            logger.error(f"Pandas query failed. Code: '{generated_code}', Error: {e}")
            return "Sorry, I was unable to process that data query."

