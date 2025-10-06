import os
from dotenv import load_dotenv
import traceback

# --- LangChain imports ---
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import Tool, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.base import BaseTool
from langchain.chains.llm import LLMChain
from langchain.agents import AgentExecutor as OrigAgentExecutor # For the custom executor
from typing import List, Dict, Any

# --- SQLAlchemy for richer query results ---
from sqlalchemy import text, create_engine
from sqlalchemy.exc import OperationalError
import time

# --- Python's Decimal type for handling database decimals ---
from decimal import Decimal

# --- Python's date and datetime types for handling database dates/timestamps ---
from datetime import date, datetime

# --- Plotting (Django integration for serving images) ---
import json
import uuid
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from django.conf import settings

load_dotenv() # Load environment variables from .env file in project root

# --- Configuration ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0))
DB_URI = os.environ.get("DATABASE_URL")

if not DB_URI:
    print("Warning: DATABASE_URL environment variable not set. SQL features will fail.")
    # raise ValueError("DATABASE_URL environment variable not set or .env file not loaded.")

# --- LLM & Database (Initialize only if DB_URI is present) ---
llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=TEMPERATURE)
db = None
if DB_URI:
    try:
        # Create engine with connection pooling safeguards
        engine = create_engine(
            DB_URI,
            pool_pre_ping=True,  # Test connections before use
            pool_recycle=300,    # Recycle connections every 5 minutes
            pool_timeout=20,     # Timeout for getting connection from pool
            max_overflow=10,     # Allow extra connections beyond pool_size
        )
        
        db = SQLDatabase(
            engine=engine,
            include_tables=[
                "categories", "customers", "customer_customer_demo", "customer_demographics",
                "employees", "employee_territories", "order_details", "orders",
                "products", "region", "shippers", "suppliers", "territories", "us_states",
            ],
            sample_rows_in_table_info=2,
        )
    except Exception as e:
        print(f"Error connecting to database: {e}")
        db = None # Ensure db is None if connection fails
else:
    print("Database URI not provided. SQL-related tools will not function.")


# --- Custom Exception for SQL Approval ---
class SQLApprovalRequired(Exception):
    def __init__(self, sql: str, tool_input: str, agent_scratchpad: str):
        super().__init__("SQL approval required")
        self.sql = sql
        self.tool_input = tool_input
        self.agent_scratchpad = agent_scratchpad

# --- SQL Chain (for SQL generation within the tool) ---
custom_sql_prompt_template = PromptTemplate(
    input_variables=["input", "table_info", "dialect", "top_k"],
    template=(
        "You are a PostgreSQL expert. Given an input question, create a syntactically correct {dialect} query to run. "
        "Pay close attention to the provided schema in the `Table DDL` section below and only use the column names listed there. "
        "Do not query for columns that do not exist. Ensure join conditions use correct column names from the schema.\n\n"
        "Only use the following tables (DDL commands follow each table name):\n{table_info}\n\n"
        "Return only the SQL query, with no markdown formatting or code fences. "
        "Unless the user explicitly asks for a limited number of results, or the question implies a small set (e.g. 'top 3'), "
        "do NOT include a LIMIT clause. If a limit is appropriate and {top_k} is provided, use at most {top_k} results.\n\n"
        "Question: {input}\n"
        "SQLQuery:"
    )
)

# This LLMChain is used by the SQLApprovalTool to generate the SQL query.
sql_generation_llm_chain = custom_sql_prompt_template | llm

# --- Plotting Function (called by PlottingTool) ---
def create_plot_django(plot_data_str: str) -> str:
    """
    Parses JSON plot data, generates a plot with Matplotlib, saves it to a
    directory served by Django (MEDIA_ROOT/plots), and returns its URL.
    """
    try:
        plot_data = json.loads(plot_data_str)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input for plot: {str(e)}"

    x_data = plot_data.get("x_data")
    y_data = plot_data.get("y_data")
    x_label = plot_data.get("x_label", "X-axis")
    y_label = plot_data.get("y_label", "Y-axis")
    title = plot_data.get("title", "Plot")
    plot_type = plot_data.get("plot_type", "line").lower()

    if not x_data or not y_data:
        return "Error: 'x_data' and 'y_data' are required for plotting."
    if len(x_data) != len(y_data):
        return "Error: 'x_data' and 'y_data' must have the same length."

    try:
        # Ensure backend is set for non-interactive environments like Django
        plt.switch_backend('Agg') # Already handled by Django's setup typically for server-side
        
        fig, ax = plt.subplots(figsize=(12, 7)) # Create figure and axes

        if plot_type == 'bar':
            ax.bar(x_data, y_data)
        elif plot_type == 'line':
            ax.plot(x_data, y_data, marker='o')
        else:
            plt.close(fig) # Close the figure if plot type is invalid
            return f"Error: Unsupported plot_type '{plot_type}'. Use 'line' or 'bar'."

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right") # Use plt for xticks if ax doesn't have it directly or for figure-level settings
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ','))) # Apply to axes
        
        fig.tight_layout() # Apply to figure

        # Ensure MEDIA_ROOT and 'plots' subdirectory exist
        # Ensure settings are configured before accessing settings.MEDIA_ROOT
        if not settings.configured:
            settings.configure() # Basic configuration if not already done by Django
            
        plots_dir = os.path.join(settings.MEDIA_ROOT, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Generate unique filename
        filename = f"plot_{uuid.uuid4().hex}.png"
        filepath = os.path.join(plots_dir, filename)
        
        fig.savefig(filepath) # Save the figure
        plt.close(fig) # Close the figure to free memory

        # Ensure MEDIA_URL ends with a slash for proper joining
        media_url = settings.MEDIA_URL
        if not media_url.endswith('/'):
            media_url += '/'
            
        plot_url = f"{media_url}plots/{filename}"
        # Standardize URL to use forward slashes, common for web URLs
        plot_url = plot_url.replace("\\", "/") 

        return f"Plot generated at: {plot_url}"

    except Exception as e:
        if 'fig' in locals() and fig is not None: # Ensure fig exists before trying to close
            plt.close(fig)
        # Log the full traceback for server-side debugging
        print(f"Error generating plot: {str(e)}\n{traceback.format_exc()}")
        return f"Error generating plot: {str(e)}"

# --- Plotting Tool (Django, as BaseTool subclass) ---
class PlottingTool(BaseTool):
    name: str = "graph_plotter"
    description: str = (
        "Use this tool to generate a visual plot (line or bar chart) when the user asks for a graph or a visual representation of data. "
        "This tool should be used AFTER you have retrieved the necessary data using the 'query_database' tool. "
        "Input MUST be a single string which is a valid JSON object representing a dictionary with the following keys: "
        "'x_data' (list of numbers or categories for x-axis), "
        "'y_data' (list of numbers for y-axis), "
        "'x_label' (string for x-axis label), "
        "'y_label' (string for y-axis label), "
        "'title' (string for plot title). "
        "Optionally, include 'plot_type' ('line' or 'bar', defaults to 'line'). "
        "The tool will return the URL of the generated plot image or an error message."
    )

    def _run(self, tool_input: str, run_manager=None):
        return create_plot_django(tool_input)

    async def _arun(self, tool_input: str, run_manager=None):
        return self._run(tool_input, run_manager)

plotting_tool = PlottingTool()

# --- Table Display Tool ---
class TableTool(BaseTool):
    name: str = "display_table"
    description: str = (
        "Use this tool to display tabular data. Input should be a JSON string with keys 'columns' (list of column names) and 'rows' (list of row lists)."
    )

    def _run(self, tool_input: str, run_manager=None):
        # Prefix the JSON for frontend detection
        return f"DATA_TABLE: {tool_input}"

    async def _arun(self, tool_input: str, run_manager=None):
        return self._run(tool_input, run_manager)

# --- Custom SQL Tool for Approval (as BaseTool subclass) ---
class SQLApprovalTool(BaseTool):
    name: str = "query_database"
    description: str = (
        "Use this tool to answer questions about the Northwind Postgres database. "
        "Input should be a concise English question. The tool will auto‑generate SQL. "
        "The SQL will be shown for approval before execution."
    )

    def _run(self, tool_input: str, run_manager=None):
        if not db:
            return "Database connection is not available. Cannot query the database."
        inputs_for_sql_gen = {
            "input": tool_input,
            "table_info": db.get_table_info(),
            "dialect": db.dialect,
            "top_k": 10, # Default top_k, can be made dynamic if needed
        }
        try:
            sql_query_message = sql_generation_llm_chain.invoke(inputs_for_sql_gen)
            sql_query_with_prefix = sql_query_message.content if hasattr(sql_query_message, 'content') else str(sql_query_message)
        except Exception as e:
            return f"Error generating SQL: {str(e)}"
        sql_query = sql_query_with_prefix.split("SQLQuery:")[-1].strip()
        raise SQLApprovalRequired(sql_query, tool_input, "")

    async def _arun(self, tool_input: str, run_manager=None):
        return self._run(tool_input, run_manager)

# --- Main tools list (all BaseTool subclasses) ---
tools = [SQLApprovalTool(), plotting_tool, TableTool()]

# --- ReAct Agent Prompt ---
# (Ensure this prompt matches the one from your working Streamlit app)
react_prompt_template_str = (
    "You are an expert business data analyst with access to a Postgres database of the Northwind sample data.\n"
    "Answer the following questions as best you can. You have access to the following tools:\n\n"
    "{tools}\n\n"
    "Use the following format for your thought process and actions:\n\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do to answer the question. This is your internal reasoning. "
    "If you need to plot data, first retrieve it using 'query_database', then use 'graph_plotter'. "
    "If you need to display tabular data (e.g., a list of results from 'query_database'), first retrieve the data using 'query_database', then use 'display_table' to format it.\n"
    "Action: the action to take, should be EXACTLY one of [{tool_names}]\n"
    "Action Input: the input to the action/tool. For 'query_database', a natural language question. "
    "For 'graph_plotter', a single string which is a valid JSON object with keys 'x_data', 'y_data', etc. "
    "For 'display_table', a single string which is a valid JSON object with keys 'columns' (list of column names) and 'rows' (list of lists of row data).\n"
    "Observation: the result of the action (this is automatically added by the system after you take an action)\n"
    "... (this Thought/Action/Action Input/Observation sequence can repeat multiple times if needed)\n\n"
    "When you have gathered enough information, your 'Final Answer' should summarize the findings. "
    "If a plot was generated, the 'Final Answer' should include both a textual summary AND the URL of the plot image provided in the observation from 'graph_plotter'. The URL should be clearly identifiable (e.g., 'Plot generated at: [URL]'). "
    "If a table was displayed using 'display_table', the 'Final Answer' should include a textual summary AND the exact `DATA_TABLE:{{...}}` string from the 'display_table' observation. This allows the table to be rendered in the chat. "
    "Do not repeat the raw data from the table in the textual summary part of the Final Answer if 'display_table' was used; simply refer to the table that was shown and include the `DATA_TABLE` string.\n"
    "Final Answer: the final answer to the original input question. If a plot was made, include its URL. If a table was displayed, include the `DATA_TABLE:{{...}}` string.\n\n"
    "Here is the conversation history:\n"
    "{chat_history}\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "{agent_scratchpad}"
)

react_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names", "chat_history"],
    template=react_prompt_template_str
)

react_agent_runnable = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

# --- Custom Agent Executor ---
class ApprovalAgentExecutor(OrigAgentExecutor):
    """Custom executor that captures the agent's scratchpad upon SQLApprovalRequired."""
    def invoke(self, inputs, **kwargs):
        # inputs is the dict like {"input": ..., "chat_history": ..., "agent_scratchpad": ...}
        # The "agent_scratchpad" key here is the accumulated scratchpad for the *current* agent run.
        current_internal_scratchpad = inputs.get("agent_scratchpad", "")
        try:
            return super().invoke(inputs, **kwargs)
        except SQLApprovalRequired as e:
            # The tool raised SQLApprovalRequired. We now set the *actual* agent scratchpad.
            e.agent_scratchpad = current_internal_scratchpad
            raise e
        except Exception as ex_other:
            # print(f"DEBUG: Other exception in ApprovalAgentExecutor. Current scratchpad: {current_internal_scratchpad}")
            raise ex_other

agent_executor = ApprovalAgentExecutor(
    agent=react_agent_runnable, # Use the runnable agent
    tools=tools,
    verbose=True, # Set to False in production if too noisy
    handle_parsing_errors=True # Or a custom error message
)

# --- Final Answer Chain (for summarizing after SQL execution) ---
final_answer_prompt_template = PromptTemplate(
    input_variables=["question", "sql", "result"],
    template=(
        "You are an expert business data analyst with access to the Northwind database. "
        "A user asked: {question}\n"
        "The following SQL query was generated and approved by the user:\n{sql}\n"
        "Executing this query returned the following result:\n{result}\n\n"
        "Based on this information, provide a concise, human-readable answer to the user's original question. "
        "Do not include any SQL or markdown formatting in your final answer."
    )
)

final_answer_chain = final_answer_prompt_template | llm

# --- Wrapper Functions for Django Views ---
def initialize_agent():
    """Called by views.py when Django app starts (or on first request)."""
    # Currently, components are initialized at module load time.
    # Add any specific one-time setup if needed.
    if not DB_URI:
        print("AGENT_LOGIC: Warning - DATABASE_URL not set. SQL features disabled.")
    elif not db:
        print("AGENT_LOGIC: Warning - Database connection failed. SQL features disabled.")
    else:
        print("AGENT_LOGIC: Agent components initialized, DB connection seems OK.")
    pass

def run_agent_invoke(prompt: str, chat_history_list_of_dicts: list, agent_scratchpad_str: str):
    """View calls this to run the agent. Catches SQLApprovalRequired."""
    if not db and any(tool.name == "query_database" for tool in tools):
         raise SQLApprovalRequired("No SQL generated - Database not connected.", prompt, agent_scratchpad_str)
        # return {"output": "I cannot answer questions requiring database access as the database is not connected."}

    # Convert chat history from view's format (list of dicts) to LangChain's format (list of BaseMessages)
    formatted_lc_history = []
    for entry in chat_history_list_of_dicts:
        content = entry.get("content", "")
        if not isinstance(content, str):
            content = str(content) # Ensure content is a string

        if entry.get("role") == "user":
            formatted_lc_history.append(HumanMessage(content=content))
        elif entry.get("role") == "assistant":
            # Avoid adding SQL approval markers or complex dicts as AIMessage content
            if "__awaiting_sql_approval__" not in content:
                formatted_lc_history.append(AIMessage(content=content))
    
    try:
        result = agent_executor.invoke({
            "input": prompt,
            "chat_history": formatted_lc_history,
            "agent_scratchpad": agent_scratchpad_str
        })
        output = result.get("output", str(result))
        # Clean up any stray characters that might appear after plot generation
        if isinstance(output, str):
            # Remove any stray single characters at the end that aren't part of URLs or meaningful text
            output = output.rstrip()
            # Remove any stray single characters that appear after plot URLs
            if "Plot generated at:" in output:
                # Split on the plot marker and clean up any stray characters
                parts = output.split("Plot generated at:")
                if len(parts) == 2:
                    before_plot = parts[0].rstrip()
                    after_plot = parts[1].strip()
                    # Remove any stray single characters after the URL
                    after_plot = after_plot.rstrip()
                    # Rejoin with proper spacing
                    if before_plot and after_plot:
                        output = f"{before_plot}\n\nPlot generated at:{after_plot}"
                    elif before_plot:
                        output = f"{before_plot}\n\nPlot generated at:{after_plot}"
                    else:
                        output = f"Plot generated at:{after_plot}"
        return {"output": output}
    except SQLApprovalRequired as e:
        # The view will catch this exception and handle UI/session state
        e.agent_scratchpad = agent_scratchpad_str # Ensure scratchpad is part of the exception
        raise e
    except Exception as e:
        # print(f"Error during agent_executor.invoke: {e}") # For debugging
        # To make it more user-friendly, you might not want to expose raw Langchain errors.
        return {"output": f"An error occurred while processing your request: {str(e)}"}

def execute_sql_query(sql_query: str) -> Dict[str, Any] | str:
    """View calls this to run the approved SQL query.
    Returns a dictionary with 'columns' and 'rows' on success, or an error string.
    """
    if not db:
        return "Database connection is not available. Cannot execute SQL."
    
    max_retries = 2
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            with db._engine.connect() as connection:
                # Sanitize or validate sql_query if it comes directly from user input elsewhere,
                # though here it's generated by the LLM and approved.
                result = connection.execute(text(sql_query))
                columns = [str(col) for col in result.keys()] # Ensure column names are strings
                rows = [[
                    item.isoformat() if isinstance(item, (date, datetime))
                    else str(item) if isinstance(item, Decimal)
                    else item
                    for item in row
                ] for row in result.fetchall()]
                connection.close()
            return {"columns": columns, "rows": rows}
        except OperationalError as e:
            if "SSL connection has been closed unexpectedly" in str(e) and attempt < max_retries - 1:
                print(f"SSL connection error on attempt {attempt + 1}: {e}. Disposing engine and retrying...")
                # Dispose the engine to clear stale connections
                db._engine.dispose()
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                return f"Error executing SQL after {max_retries} attempts: {str(e)}"
        except Exception as e:
            # print(f"Error during db.run: {e}") # For debugging
            return f"Error executing SQL: {str(e)}"

def run_final_chain(question: str, sql: str, result: str):
    """View calls this to get the final summarized answer after SQL execution."""
    try:
        response_message = final_answer_chain.invoke({
            "question": question,
            "sql": sql,
            "result": str(result) # Ensure result is a string
        })
        return response_message.content if hasattr(response_message, 'content') else str(response_message)
    except Exception as e:
        # print(f"Error in final_answer_chain: {e}") # For debugging
        return f"Error generating final answer: {str(e)}"

# --- Initial check --- 
# (This runs when the module is first imported by Django)
if __name__ == '__main__':
    # This block is for testing agent_logic.py directly, not used by Django runserver
    print("Testing agent_logic.py setup...")
    initialize_agent()
    if db:
        print(f"Connected to DB: {db.dialect} - Sample tables: {db.get_usable_table_names()[:3]}")
    else:
        print("DB connection not established during test.")
    # Test SQL generation (will raise SQLApprovalRequired)
    # try:
    #     test_result = run_agent_invoke("How many customers are there?", [], "")
    #     print(f"Test agent invoke (no approval): {test_result}")
    # except SQLApprovalRequired as e:
    #     print(f"SQLApprovalRequired caught: SQL='{e.sql}', Input='{e.tool_input}'")
    #     # Test final chain
    #     final_ans = run_final_chain("How many customers?", e.sql, "[(91,)]")
    #     print(f"Test final chain: {final_ans}")
    # except Exception as e:
    #     print(f"Error during direct test: {e}")

# ----------------- Helper: auto-execute query_database (no approval) --------------

def _auto_execute_query(tool_input: str) -> str:
    """Turn NL question -> SQL (using same prompt), execute immediately, return result."""
    if not db:
        return "Database connection not available."
    
    max_retries = 2
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            inputs_for_sql_gen = {
                "input": tool_input,
                "table_info": db.get_table_info(),
                "dialect": db.dialect,
                "top_k": 10,
            }
            msg = sql_generation_llm_chain.invoke(inputs_for_sql_gen)
            sql_query_with_prefix = msg.content if hasattr(msg, "content") else str(msg)
            sql_query = sql_query_with_prefix.split("SQLQuery:")[-1].strip()
            result = db.run(sql_query)
            return str(result)
        except OperationalError as e:
            if "SSL connection has been closed unexpectedly" in str(e) and attempt < max_retries - 1:
                print(f"SSL connection error in auto-execute on attempt {attempt + 1}: {e}. Disposing engine and retrying...")
                # Dispose the engine to clear stale connections
                db._engine.dispose()
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                return f"Error running auto SQL after {max_retries} attempts: {exc}"
        except Exception as exc:
            return f"Error running auto SQL: {exc}"


def get_auto_execute_tools() -> List[BaseTool]:
    """Return tools list where `query_database` runs immediately without approval."""
    class AutoQueryDBTool(BaseTool):
        name: str = "query_database"
        description: str = (
            "Use this tool to answer questions about the Northwind Postgres database. "
            "Input should be a concise English question. The tool will auto‑generate SQL. "
            "The SQL will be shown for approval before execution."
        )

        def _run(self, tool_input: str, run_manager=None):
            return _auto_execute_query(tool_input)

        async def _arun(self, tool_input: str, run_manager=None):
            return self._run(tool_input, run_manager)

    return [AutoQueryDBTool(), plotting_tool, TableTool()]


# Generic helper to run the agent with a supplied tools list (used for resume step)
def run_agent_invoke_with_tools(prompt: str, chat_history_list_of_dicts: list, agent_scratchpad_str: str, custom_tools: List[Tool]):
    """Invoke a *fresh* agent executor built with `custom_tools`."""
    
    try:
        # Create the agent runnable with tools correctly bound for the prompt
        temp_agent_runnable = create_react_agent(
            llm=llm, 
            tools=custom_tools, # Pass tools here for the agent to use
            prompt=react_prompt # The original prompt template is used
        )
        
        temp_executor = OrigAgentExecutor(
            agent=temp_agent_runnable,
            tools=custom_tools, # Also pass tools to the executor
            verbose=True,
            handle_parsing_errors=True,
        )

        # Convert history to LangChain messages
        formatted_lc_history = []
        for entry in chat_history_list_of_dicts:
            content = entry.get("content", "")
            if entry.get("role") == "user":
                formatted_lc_history.append(HumanMessage(content=content))
            elif entry.get("role") == "assistant":
                # Ensure AIMessage content is a string, not a dict (like __awaiting_sql_approval__)
                if isinstance(content, str):
                     formatted_lc_history.append(AIMessage(content=content))
                # else:
                    # print(f"Skipping non-string AIMessage content from history: {content}")

        # The input to invoke for a ReAct agent typically includes "input" and "agent_scratchpad",
        # and "chat_history". "tools" and "tool_names" are usually for the prompt formatting phase.
        # create_react_agent internally handles the prompt formatting with tools.
        input_dict = {
            "input": prompt,
            "chat_history": formatted_lc_history,
            "agent_scratchpad": agent_scratchpad_str,
            # If create_react_agent doesn't bake these into the runnable's partial_vars,
            # they might still be needed if the prompt template itself wasn't fully resolved.
            # However, create_react_agent is supposed to handle this.
            # Let's rely on create_react_agent to format the prompt with tools.
        }
        
        result = temp_executor.invoke(input_dict)
        output = result.get("output", str(result))
        # Clean up any stray characters that might appear after plot generation
        if isinstance(output, str):
            # Remove any stray single characters at the end that aren't part of URLs or meaningful text
            output = output.rstrip()
            # Remove any stray single characters that appear after plot URLs
            if "Plot generated at:" in output:
                # Split on the plot marker and clean up any stray characters
                parts = output.split("Plot generated at:")
                if len(parts) == 2:
                    before_plot = parts[0].rstrip()
                    after_plot = parts[1].strip()
                    # Remove any stray single characters after the URL
                    after_plot = after_plot.rstrip()
                    # Rejoin with proper spacing
                    if before_plot and after_plot:
                        output = f"{before_plot}\n\nPlot generated at:{after_plot}"
                    elif before_plot:
                        output = f"{before_plot}\n\nPlot generated at:{after_plot}"
                    else:
                        output = f"Plot generated at:{after_plot}"
        return {"output": output}

    except Exception as exc:
        error_info = f"Error in custom agent invoke: {type(exc).__name__} - {str(exc)}"
        print(f"TRACEBACK for custom agent invoke error: {traceback.format_exc()}") # Print traceback for better debugging
        return {"output": error_info}
