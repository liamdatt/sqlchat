# Install deps:
#   pip install -U streamlit python-dotenv sqlalchemy psycopg2-binary langchain-core langchain-community langchain-experimental langchain-openai

import os
import streamlit as st
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import uuid
import json

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0))
DB_URI = os.environ["DATABASE_URL"]  # required


llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=TEMPERATURE)

db = SQLDatabase.from_uri(
    DB_URI,
    include_tables=[
        "categories", "customers", "customer_customer_demo", "customer_demographics",
        "employees", "employee_territories", "order_details", "orders",
        "products", "region", "shippers", "suppliers", "territories", "us_states",
    ],
    sample_rows_in_table_info=2,
)

# Create a custom SQL prompt that forbids markdown and LIMIT clauses
custom_sql_prompt = PromptTemplate(
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

# Use the custom prompt when creating the SQL chain
sql_chain = SQLDatabaseChain.from_llm(
    llm, db, prompt=custom_sql_prompt, verbose=True, return_direct=True, input_key="input"
)

def query_db(question: str) -> str:
    """Turn natural‑language into SQL via LangChain, execute, and return the result string."""
    return sql_chain.invoke({"input": question})["result"]

query_tool = Tool(
    name="query_database",
    func=query_db,
    description=(
        "Use this tool to answer questions about the Northwind Postgres database. "
        "Input should be a concise English question. The tool will auto‑generate SQL, run it, "
        "and return the result set as a string. This tool is for data retrieval only."
    ),
)

# Plotting Function and Tool
def create_plot(plot_data_str: str) -> str:
    """
    Generates a plot from the provided JSON string data and saves it as a PNG file.

    Args:
        plot_data_str (str): A JSON string representing a dictionary containing:
            'x_data': List of x-axis values.
            'y_data': List of y-axis values.
            'x_label': Label for the x-axis.
            'y_label': Label for the y-axis.
            'title': Title of the plot.
            'plot_type': Type of plot (e.g., 'line', 'bar'). Defaults to 'line'.
    Returns:
        str: The filepath of the generated plot image, or an error message.
    """
    try:
        plot_data = json.loads(plot_data_str)

        x_data = plot_data['x_data']
        y_data = plot_data['y_data']
        x_label = plot_data['x_label']
        y_label = plot_data['y_label']
        title = plot_data['title']
        plot_type = plot_data.get('plot_type', 'line').lower()

        # Ensure x_data are integers if they represent years, and y_data are floats
        x_data = [int(float(x)) if isinstance(x, (str, float, int)) and str(x).replace('.','',1).isdigit() else x for x in x_data]
        y_data = [float(y) for y in y_data]

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 7))

        if plot_type == 'bar':
            bars = plt.bar(x_data, y_data, color='skyblue', edgecolor='black')
        elif plot_type == 'line':
            plt.plot(x_data, y_data, marker='o', linestyle='-', color='royalblue')
        else:
            return f"Error: Unsupported plot_type '{plot_type}'. Supported types are 'line', 'bar'."

        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Format Y-axis as currency
        formatter = mticker.FormatStrFormatter('$%.0f')
        plt.gca().yaxis.set_major_formatter(formatter)
        
        # Ensure x-axis ticks are integers if they are years
        if all(isinstance(x, int) for x in x_data):
             plt.xticks(x_data, [str(x) for x in x_data], rotation=45, ha="right", fontsize=10)
        else:
            plt.xticks(rotation=45, ha="right", fontsize=10)
        
        plt.yticks(fontsize=10)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if not os.path.exists("plots"):
            os.makedirs("plots")

        filename = f"plots/plot_{uuid.uuid4()}.png"
        plt.savefig(filename)
        plt.close()
        return f"Plot generated and saved to: {filename}"
    except Exception as e:
        # Print detailed error to console for debugging
        import traceback
        print(f"Error in create_plot: {e}")
        traceback.print_exc()
        return f"Error generating plot: {str(e)}. Check data format and types."

plotting_tool = Tool(
    name="graph_plotter",
    func=create_plot,
    description=(
        "Use this tool to generate a visual plot (line or bar chart) when the user asks for a graph or a visual representation of data. "
        "This tool should be used AFTER you have retrieved the necessary data using the 'query_database' tool. "
        "Input MUST be a single string which is a valid JSON object representing a dictionary with the following keys: "
        "'x_data' (list of numbers or categories for x-axis), "
        "'y_data' (list of numbers for y-axis), "
        "'x_label' (string for x-axis label), "
        "'y_label' (string for y-axis label), "
        "'title' (string for plot title). "
        "Optionally, include 'plot_type' ('line' or 'bar', defaults to 'line'). "
        "The tool will return the filepath of the generated plot image or an error message."
    )
)

# ReAct agent (v0.2 style)
tools = [query_tool, plotting_tool] # Add new tool here

prompt_template_str = (
    "You are an expert business data analyst with access to a Postgres database of the Northwind sample data.\\n"
    "Answer the following questions as best you can. You have access to the following tools:\\n\\n"
    "{tools}\\n\\n" # This will now include the graph_plotter
    "Use the following format for your thought process and actions:\\n\\n"
    "Question: the input question you must answer\\n"
    "Thought: you should always think about what to do to answer the question. This is your internal reasoning. If you need to plot data, first retrieve it using 'query_database', then use 'graph_plotter'.\\n"
    "Action: the action to take, should be EXACTLY one of [{tool_names}]\\n" # This will now include graph_plotter
    "Action Input: the input to the action/tool. For 'query_database', a natural language question. For 'graph_plotter', a single string which is a valid JSON object representing a dictionary with the following keys: "
    "'x_data' (list of numbers or categories for x-axis), "
    "'y_data' (list of numbers for y-axis), "
    "'x_label' (string for x-axis label), "
    "'y_label' (string for y-axis label), "
    "'title' (string for plot title). "
    "Optionally, include 'plot_type' ('line' or 'bar', defaults to 'line').\\n"
    "Observation: the result of the action (this is automatically added by the system after you take an action)\\n"
    "... (this Thought/Action/Action Input/Observation sequence can repeat multiple times if needed)\\n\\n"
    "IMPORTANT: If an Action (like a database query) returns partial or limited results, and you determine that more data is needed to answer the user's question (based on the observation or chat history), your next Thought should be to re-query for the complete data. Your subsequent Action Input should then be a more specific question to the tool that requests all necessary data (e.g., by asking for 'all results' or 'N results' if N is known). Do not assume you have all data if the observation indicates a limit.\\n\\n"
    "Also, when constructing your Action Input for tools like the database:\\n"
    "- Do NOT include any markdown formatting or code fences (e.g., ```sql). Use plain natural language.\\n"
    "- To request the entire dataset, explicitly state 'return all rows, do not use LIMIT clause'.\\n\\n"
    "When using 'graph_plotter':\\n"
    "- Ensure 'x_data' and 'y_data' are lists of appropriate values extracted from previous observations (e.g., database query results).\\n"
    "- Provide clear 'x_label', 'y_label', and 'title' for the plot.\\n\\n"
    "When you have gathered enough information, and if a plot was generated, the 'Final Answer' should include both a textual summary AND the filepath of the plot image provided in the observation from 'graph_plotter'. "
    "The filepath should be prefixed with \"Plot generated: \" (e.g., 'Plot generated: plots/plot_xyz.png').\\n"
    "Final Answer: the final answer to the original input question. If a plot was made, mention it and include its path using the format: 'Plot generated: plots/plot_abc.png'.\\n\\n"
    "Here is the conversation history:\\n"
    "{chat_history}\\n\\n"
    "Begin!\\n\\n"
    "Question: {input}\\n"
    "{agent_scratchpad}"
)

prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names", "chat_history"],
    template=prompt_template_str
)

react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)
agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Streamlit UI
st.set_page_config(page_title="Northwind AI Data Analyst", page_icon="📊", layout="wide")

st.title("📊 SQL AI Data Analyst")
with st.expander("ℹ️ How it works", expanded=False):
    st.markdown(
        """
        **Ask anything about the Northwind database!**  
        Your question → LLM plans → SQL is generated & executed → answer streams back.
        """
    )

if "history" not in st.session_state:
    st.session_state.history = []  # [(role, content)]

prompt = st.chat_input("Ask a business question…")

if prompt:
    st.session_state.history.append(("user", prompt))
    with st.spinner("🤖 Thinking…"):
        try:
            # Format chat history for the agent
            formatted_history = []
            for role, content in st.session_state.history[:-1]: # Exclude current prompt
                if role == "user":
                    formatted_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    formatted_history.append(AIMessage(content=content))

            result = agent_executor.invoke({
                "input": prompt,
                "chat_history": formatted_history
            })
            answer = result.get("output", str(result))
        except Exception as e:
            answer = f"⚠️ Error: {e}"
    st.session_state.history.append(("assistant", answer))

# Render chat history
for role, content in st.session_state.history:
    msg = st.chat_message(role)
    if role == "assistant":
        plot_marker = "Plot generated: "
        if plot_marker in content and ".png" in content:
            try:
                parts = content.split(plot_marker)
                text_before_plot = parts[0].strip()
                
                # The rest of the string after the marker might contain the path and then more text
                remaining_after_marker = parts[1]
                plot_path_end_index = remaining_after_marker.find(".png") + 4
                plot_path = remaining_after_marker[:plot_path_end_index].strip()
                text_after_plot = remaining_after_marker[plot_path_end_index:].strip()

                if text_before_plot:
                    msg.write(text_before_plot)
                
                if os.path.exists(plot_path):
                    msg.image(plot_path)
                else:
                    msg.write(f"\n(Plot image not found at {plot_path})\n")
                
                if text_after_plot:
                    msg.write(text_after_plot)

            except Exception as e:
                msg.write(content) # Fallback
                msg.error(f"Error parsing/displaying plot message: {e}")
        elif "\t" in content or content.strip().startswith("|"): # Handle tables
            msg.table(content)
        else:
            msg.write(content)
    else: # User message
        msg.write(content)

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    st.text_input("OpenAI model", value=OPENAI_MODEL, key="model_name")
    st.slider("Temperature", 0.0, 1.0, TEMPERATURE, 0.01, key="temp")
    st.markdown("---")
    st.markdown(
        "Built with **Streamlit**, **LangChain v0.2** and **OpenAI** on the Northwind Postgres sample database."
    )
