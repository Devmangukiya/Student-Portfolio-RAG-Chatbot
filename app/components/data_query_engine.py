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


# --- THE CRITICAL FIX (Part 1): An Upgraded, More Robust Prompt ---
# We add a new rule and a new example to teach the LLM how to handle
# queries that do not have a filtering condition.
PROMPT_STRUCTURED_QUERY = PromptTemplate.from_template(
    """
### ROLE ###
You are an expert data analyst. Your task is to parse a user's natural language question and extract key filtering criteria into a structured JSON format.

### CONTEXT ###
You are working with a dataset of student achievements with columns like: ['name', 'email', 'student_id', 'department', 'type', 'status'].

### TASK ###
Based on the user's question, identify the column to filter on, the value to filter by, and the column to return.

### IMPORTANT RULES ###
1. Your output MUST be ONLY a valid JSON object.
2. If the user asks for ALL items without any filtering condition (e.g., "list all students"), set `column_to_filter` and `filter_value` to `null`.
3. The `filter_value` should be the specific text the user is searching for, converted to lowercase.
4. The `column_to_return` is the information the user wants listed (e.g., 'name').

### EXAMPLES ###
User Question: "give me list of all students which have workshop type achievement"
{{"column_to_filter": "type", "filter_value": "workshop", "column_to_return": "name"}}

User Question: "give me the name of all students"
{{"column_to_filter": null, "filter_value": null, "column_to_return": "name"}}

### USER QUESTION ###
{question}

### JSON OUTPUT ###
"""
)

class DataQueryEngine:
    def __init__(self):
        try:
            logger.info("Initializing DataQueryEngine with structured output...")
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
            self.structured_query_chain = PROMPT_STRUCTURED_QUERY | llm | StrOutputParser()
            
        except Exception as e:
            raise CustomException(f"Failed to initialize DataQueryEngine: {e}")

    def query_data(self, user_query: str) -> str:
        try:
            logger.info(f"Generating structured query for: '{user_query}'")
            json_response = self.structured_query_chain.invoke({"question": user_query})
            query_params = json.loads(json_response)
            
            col_filter = query_params.get('column_to_filter')
            val_filter = query_params.get('filter_value')
            col_return = query_params.get('column_to_return')

            if not col_return:
                raise ValueError("'column_to_return' is missing from the LLM's response.")

            # --- THE CRITICAL FIX (Part 2): Handle the "No Filter" Case ---
            # This new logic checks if a filter was provided. If not, it returns
            # all unique values from the requested column.
            if col_filter and val_filter is not None:
                logger.info(f"Executing FILTERED query: filter '{col_filter}' on '{val_filter}'")
                result_series = self.df[self.df[col_filter] == val_filter.lower()][col_return]
            else:
                logger.info(f"Executing UNFILTERED query: returning all unique values for '{col_return}'")
                result_series = self.df[col_return]

            result = result_series.unique().tolist()
            
            if not result:
                return "No results found for that query."
            
            formatted_result = [f"{i+1}. {str(item).title()}" for i, item in enumerate(result)]
            return "Here are the results:\n" + "\n".join(formatted_result)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse or use LLM's structured output. Response: '{json_response}'. Error: {e}", exc_info=True)
            return "Sorry, I had trouble understanding the structure of that query."
        except Exception as e:
            logger.error(f"An unexpected error occurred during data query: {e}", exc_info=True)
            return "Sorry, I was unable to process that data query."

