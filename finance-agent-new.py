import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import pandas as pd
from io import StringIO
from contextlib import contextmanager

@contextmanager
def get_memory():
    with SqliteSaver.from_conn_string(":memory:") as memory:
        yield memory

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

llm_name = "gpt-4o-mini"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

class AgentState(TypedDict):
    task: str
    csv_file: str
    financial_data: str
    analysis: str
    report: str
    initiatives: str

# Define prompts
GATHER_FINANCIALS_PROMPT = """You are an expert financial analyst. Analyze the given financial data and provide a detailed summary."""
ANALYZE_DATA_PROMPT = """You are an expert financial analyst. Based on the financial data summary, provide a comprehensive analysis including key financial metrics, trends, and insights."""
WRITE_REPORT_PROMPT = """You are a financial report writer. Write a detailed, comprehensive financial report based on the analysis provided. Include sections on financial health, performance metrics, risk assessment, and future outlook."""
GENERATE_INITIATIVES_PROMPT = """You are a strategic financial consultant. Based on the financial report and analysis provided, generate a list of actionable initiatives to lower costs and improve financial performance. Also, provide additional tips for financial optimization. Be specific and practical in your recommendations."""

def gather_financials_node(state: AgentState):
    csv_file = state["csv_file"]
    df = pd.read_csv(StringIO(csv_file))
    financial_data_str = df.to_string(index=False)
    combined_content = f"{state['task']}\n\nHere is the financial data:\n\n{financial_data_str}"
    messages = [
        SystemMessage(content=GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=combined_content),
    ]
    response = model.invoke(messages)
    return {"financial_data": response.content}

def analyze_data_node(state: AgentState):
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=state["financial_data"]),
    ]
    response = model.invoke(messages)
    return {"analysis": response.content}

def write_report_node(state: AgentState):
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=state["analysis"]),
    ]
    response = model.invoke(messages)
    return {"report": response.content}

def generate_initiatives_node(state: AgentState):
    messages = [
        SystemMessage(content=GENERATE_INITIATIVES_PROMPT),
        HumanMessage(content=f"Financial Analysis:\n{state['analysis']}\n\nFinancial Report:\n{state['report']}"),
    ]
    response = model.invoke(messages)
    return {"initiatives": response.content}

builder = StateGraph(AgentState)

builder.add_node("gather_financials", gather_financials_node)
builder.add_node("analyze_data", analyze_data_node)
builder.add_node("write_report", write_report_node)
builder.add_node("generate_initiatives", generate_initiatives_node)

builder.set_entry_point("gather_financials")

builder.add_edge("gather_financials", "analyze_data")
builder.add_edge("analyze_data", "write_report")
builder.add_edge("write_report", "generate_initiatives")

with get_memory() as memory:
    graph = builder.compile(checkpointer=memory)

# Streamlit UI
import streamlit as st

def main():
    st.title("In-Depth Financial Analysis Report Generator")

    task = st.text_input(
        "Enter the task:",
        "Provide a comprehensive financial analysis of our company NaikiAI",
        key="task_input"
    )
    uploaded_file = st.file_uploader(
        "Upload a CSV file with the company's financial data", 
        type=["csv"],
        key="file_uploader"
    )

    if st.button("Generate Report", key="generate_button") and uploaded_file is not None:
        csv_data = uploaded_file.getvalue().decode("utf-8")

        initial_state = {
            "task": task,
            "csv_file": csv_data,
        }
        thread = {"configurable": {"thread_id": "1"}}

        steps = []

        with st.spinner("Generating report..."):
            with get_memory() as memory:
                graph = builder.compile(checkpointer=memory)
                for s in graph.stream(initial_state, thread):
                    steps.append(s)

        # Display each step and its context
        for step in steps:
            if isinstance(step, dict):
                step_name = list(step.keys())[0]
                step_content = step[step_name]
                
                st.subheader(f"Step: {step_name}")
                
                if step_name == "gather_financials":
                    st.write("Financial Data Summary:")
                    st.write(step_content["financial_data"])
                elif step_name == "analyze_data":
                    st.write("Analysis:")
                    st.write(step_content["analysis"])
                elif step_name == "write_report":
                    st.write("Report:")
                    st.write(step_content["report"])
                elif step_name == "generate_initiatives":
                    st.write("Initiatives:")
                    st.write(step_content["initiatives"])

        # Display final report and initiatives
        final_state = steps[-1] if steps else None
        if final_state and "generate_initiatives" in final_state:
            final_state = final_state["generate_initiatives"]

            if "report" in final_state:
                st.subheader("Final Financial Analysis Report")
                st.write(final_state["report"])

            if "initiatives" in final_state:
                st.subheader("Final Actionable Initiatives and Tips")
                st.write(final_state["initiatives"])

            

if __name__ == "__main__":
    main()