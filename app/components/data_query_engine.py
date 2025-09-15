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


PROMPT_QUERY_TYPE_ROUTER = PromptTemplate.from_template(
    """
### ROLE ###
You are an expert query analyst. Your task is to classify a user's question as either a 'simple_filter' or a 'complex_analysis'.

### DEFINITIONS ###
- `simple_filter`: A query that involves filtering text columns based on an exact match (e.g., department is 'chemistry', type is 'workshop'). This also includes requests for all items in a column (e.g. "list all students").
- `complex_analysis`: A query that involves numerical operations (count, sum, average, > <), sorting by value (highest/lowest), or any other more complex analysis that requires flexible code.

### TASK ###
Based on the user's question, respond with ONLY the classification: `simple_filter` or `complex_analysis`.

### EXAMPLES ###
User Question: "give me list of all students which have workshop type achievement" -> simple_filter
User Question: "give me name of all students" -> simple_filter
User Question: "how many students are in the chemistry department?" -> complex_analysis
User Question: "list achievements that awarded more than 3 credits" -> complex_analysis
User Question: "give me student which has total highest credit awarded" -> complex_analysis

### USER QUESTION ###
{question}

### CLASSIFICATION ###
"""
)

PROMPT_STRUCTURED_QUERY = PromptTemplate.from_template(
    """
### ROLE ###
You are an expert data analyst. Your task is to parse a user's question and extract filtering criteria into a structured JSON format.

### CONTEXT ###
You are working with a dataset of student achievements with columns like: ['name', 'department', 'type', 'status'].

### TASK ###
Based on the user's question, identify the column to filter on, the value to filter by, and the column to return.

### IMPORTANT RULES ###
1. Your output MUST be ONLY a valid JSON object.
2. If the user asks for ALL items in a column (e.g., "list all students"), set `column_to_filter` and `filter_value` to `null`.
3. The `filter_value` should be the specific text the user is searching for, converted to lowercase.
4. The `column_to_return` is the information the user wants listed.

### USER QUESTION ###
{question}

### JSON OUTPUT ###
"""
)

PROMPT_ANALYTICAL_PLAN = PromptTemplate.from_template(
    """
### ROLE ###
You are an expert data analyst. Your task is to parse a user's complex analytical question and create a structured JSON plan to answer it.

### CONTEXT ###
You are working with a dataset with columns: ['name', 'department', 'type', 'status', 'approved_by', 'credit_awarded'].

### TASK ###
Based on the user's question, create a JSON plan with the following keys:
- `groupby_col`: The column to group the data by.
- `agg_col`: The column to perform the calculation on.
- `agg_func`: The function to apply ('sum', 'count', 'idxmax').
- `sort_ascending`: boolean, true for ascending, false for descending.
- `top_n`: The number of top results to return (e.g., 10).
- `column_to_return`: The final column the user wants to see.

### IMPORTANT RULES ###
1. Your output MUST be ONLY a valid JSON object.
2. If a key is not applicable, set its value to `null`.
3. Infer the user's intent. "Most" or "highest" implies `idxmax` or sorting descending and taking the top 1.

### EXAMPLES ###
User Question: "Give me top 10 students name based on total credit points"
{{"groupby_col": "name", "agg_col": "credit_awarded", "agg_func": "sum", "sort_ascending": false, "top_n": 10, "column_to_return": "name"}}

User Question: "Give me a faculty name which approved most certificate"
{{"groupby_col": "approved_by", "agg_col": "approved_by", "agg_func": "count", "sort_ascending": false, "top_n": 1, "column_to_return": "approved_by"}}

### USER QUESTION ###
{question}

### JSON PLAN ###
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
            
            self.df['name'] = self.df['name'].str.lower()
            self.df['department'] = self.df['department'].str.lower()
            self.df['type'] = self.df['type'].str.lower()
            self.df['status'] = self.df['status'].str.lower()
            self.df['approved_by'] = self.df['approved_by'].str.lower()
            logger.info("Student data loaded and sanitized successfully.")
            
            llm = load_llm(model_name=GROQ_MODEL_NAME, groq_api_key=GROQ_API_KEY)
            self.router_chain = PROMPT_QUERY_TYPE_ROUTER | llm | StrOutputParser()
            self.structured_query_chain = PROMPT_STRUCTURED_QUERY | llm | StrOutputParser()
            self.analytical_planner_chain = PROMPT_ANALYTICAL_PLAN | llm | StrOutputParser()
        except Exception as e:
            raise CustomException(f"Failed to initialize DataQueryEngine: {e}")

    def query_data(self, user_query: str) -> str:
        try:
            logger.info(f"Classifying query: '{user_query}'")
            query_type = self.router_chain.invoke({"question": user_query}).strip()
            logger.info(f"Determined query type: '{query_type}'")

            if "simple_filter" in query_type:
                return self._execute_structured_query(user_query)
            else:
                return self._execute_analytical_query(user_query)
        except Exception as e:
            logger.error(f"Error in query_data router: {e}", exc_info=True)
            return "Sorry, I encountered an error while processing your query."

    def _execute_structured_query(self, user_query: str) -> str:
        json_response = ""
        try:
            logger.info("Executing structured (JSON) query path...")
            json_response = self.structured_query_chain.invoke({"question": user_query})
            query_params = json.loads(json_response)
            col_filter = query_params.get('column_to_filter')
            val_filter = query_params.get('filter_value')
            col_return = query_params.get('column_to_return')
            if not col_return: raise ValueError("'column_to_return' is missing.")
            
            if col_filter and val_filter is not None:
                result_series = self.df[self.df[col_filter] == val_filter.lower()][col_return]
            else:
                result_series = self.df[col_return]
            result = result_series.unique().tolist()
            
            if not result: return "No results found for that query."
            
            formatted_result = [f"{i+1}. {str(item).title()}" for i, item in enumerate(sorted(result))]
            return "Here are the results:\n" + "\n".join(formatted_result)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Structured query failed. Response: '{json_response}'. Error: {e}")
            return "Sorry, I had trouble understanding the structure of that query."
            
    def _execute_analytical_query(self, user_query: str) -> str:
        json_plan_response = ""
        try:
            logger.info("Executing complex (Analytical) query path...")
            json_plan_response = self.analytical_planner_chain.invoke({"question": user_query})
            plan = json.loads(json_plan_response)
            
            logger.info(f"Executing analytical plan: {plan}")
            
            
            df = self.df
            
            grouped_data = df.groupby(plan['groupby_col'])[plan['agg_col']]
            
            if plan['agg_func'] == 'sum':
                aggregated_data = grouped_data.sum()
            elif plan['agg_func'] == 'count':
                aggregated_data = grouped_data.size() 
            elif plan['agg_func'] == 'idxmax':
                result = grouped_data.sum().idxmax()
                return f"The result is: {str(result).title()}"
            else:
                raise ValueError(f"Unsupported aggregation function: {plan['agg_func']}")

            sorted_data = aggregated_data.sort_values(ascending=plan.get('sort_ascending', False))
            top_results = sorted_data.head(plan.get('top_n', 10))
            
            result = top_results.index.tolist()

            if not result: return "No results found for that query."
            
            formatted_result = [f"{i+1}. {str(item).title()}" for i, item in enumerate(result)]
            return "Here are the results:\n" + "\n".join(formatted_result)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Analytical query failed. Plan: '{json_plan_response}'. Error: {e}")
            return "Sorry, I had trouble creating a plan for that analytical query."
        except Exception as e:
            logger.error(f"An unexpected error occurred during analytical query: {e}")
            return "Sorry, I was unable to process that data query."

