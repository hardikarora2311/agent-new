import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import pandas as pd
from io import StringIO
from contextlib import contextmanager
from pinecone import Pinecone
# from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st


# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize MixedBread embeddings
model_name = "mixedbread-ai/mxbai-embed-large-v1"
hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

vectorstore = PineconeVectorStore(index=index, embedding=hf_embeddings, text_key="text")

@contextmanager
def get_memory():
    with SqliteSaver.from_conn_string(":memory:") as memory:
        yield memory

llm_name = "gpt-4o-mini"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

class AgentState(TypedDict):
    task: str
    csv_file: str
    financial_data: str
    analysis: str
    report: str
    initiatives: str
    context: str

# Define prompts
GATHER_FINANCIALS_PROMPT = """You are an expert financial analyst. Analyze the given financial data and provide a detailed summary. Use the provided context to enhance your analysis."""
ANALYZE_DATA_PROMPT = """You are an expert financial analyst. Based on the financial data summary and the provided context, provide a comprehensive analysis including key financial metrics, trends, and insights."""
WRITE_REPORT_PROMPT = """You are a financial report writer. Write a detailed, comprehensive financial report based on the analysis provided and the context. Include sections on financial health, performance metrics, risk assessment, and future outlook."""
GENERATE_INITIATIVES_PROMPT = """You are a strategic financial consultant. Based on the financial report, analysis provided, and the context, generate a list of actionable initiatives to lower costs and improve financial performance. Also, provide additional tips for financial optimization. Be specific and practical in your recommendations."""

def retrieve_context(query, k=3, namespace="testspace"):
    print(f"Retrieving context for query: {query}")
    try:
        # Generate embedding for the query
        query_embedding = hf_embeddings.embed_query(query)
        
        # Query the Pinecone index
        query_response = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            namespace=namespace
        )
        
        print(f"Number of results retrieved: {len(query_response['matches'])}")
        
        if len(query_response['matches']) > 0:
            context = "\n\n".join([match['metadata']['chunk'] for match in query_response['matches'] if 'chunk' in match['metadata']])
            print(f"Context retrieved (first 100 characters): {context[:100]}...")
        else:
            print("No results found in the vector store.")
            context = ""
        
        return context
    except Exception as e:
        print(f"Error retrieving context: {str(e)}")
        return ""

def check_pinecone_index():
    try:
        stats = index.describe_index_stats()
        print(f"Pinecone index stats: {stats}")
        if stats['total_vector_count'] == 0:
            print("Warning: Your Pinecone index is empty. Make sure you've ingested your documents.")
        else:
            print(f"Your Pinecone index contains {stats['total_vector_count']} vectors.")
    except Exception as e:
        print(f"Error checking Pinecone index: {str(e)}")

def inspect_pinecone_index():
    try:
        # Fetch a few vectors from the index
        fetch_response = index.fetch(ids=["1", "2", "3"])  # Adjust IDs as necessary
        print("Sample vectors from Pinecone index:")
        for id, vector in fetch_response['vectors'].items():
            print(f"ID: {id}")
            print(f"Metadata: {vector['metadata']}")
            print("---")
        
        # Perform a sample query
        query_embedding = hf_embeddings.embed_query("financial analysis")
        query_response = index.query(vector=query_embedding, top_k=1, include_metadata=True)
        print("\nSample query result:")
        print(query_response)
    except Exception as e:
        print(f"Error inspecting Pinecone index: {str(e)}")

def gather_financials_node(state: AgentState):
    csv_file = state["csv_file"]
    df = pd.read_csv(StringIO(csv_file))
    financial_data_str = df.to_string(index=False)
    context = retrieve_context(state['task'])
    print(f"gather_financials_node context (first 100 characters): {context[:100]}...")
    combined_content = f"{state['task']}\n\nHere is the financial data:\n\n{financial_data_str}\n\nContext:\n{context}"
    messages = [
        SystemMessage(content=GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=combined_content),
    ]
    response = model.invoke(messages)
    return {"financial_data": response.content, "context": context}

def analyze_data_node(state: AgentState):
    context = retrieve_context(state['financial_data'])
    print(f"analyze_data_node context (first 100 characters): {context[:100]}...")
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=f"Financial Data:\n{state['financial_data']}\n\nContext:\n{context}"),
    ]
    response = model.invoke(messages)
    return {"analysis": response.content, "context": context}

def write_report_node(state: AgentState):
    context = retrieve_context(state['analysis'])
    print(f"write_report_node context (first 100 characters): {context[:100]}...")
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=f"Analysis:\n{state['analysis']}\n\nContext:\n{context}"),
    ]
    response = model.invoke(messages)
    return {"report": response.content, "context": context}

def generate_initiatives_node(state: AgentState):
    context = retrieve_context(state['report'])
    print(f"generate_initiatives_node context (first 100 characters): {context[:100]}...")
    messages = [
        SystemMessage(content=GENERATE_INITIATIVES_PROMPT),
        HumanMessage(content=f"Financial Analysis:\n{state['analysis']}\n\nFinancial Report:\n{state['report']}\n\nContext:\n{context}"),
    ]
    response = model.invoke(messages)
    return {"initiatives": response.content, "context": context}

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

def main():
    st.title("In-Depth Financial Analysis Report Generator")
    check_pinecone_index()
    inspect_pinecone_index()
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
                    print(f"Step received: {s}")
                    steps.append(s)

        # Display each step and its context
        for step in steps:
            if isinstance(step, dict):
                step_name = list(step.keys())[0]
                step_content = step[step_name]
                
                st.subheader(f"Step: {step_name}")
                print(f"Processing step: {step_name}")
                
                if "context" in step_content:
                    st.write("Context:")
                    context_preview = step_content["context"][:500] + "..." if len(step_content["context"]) > 500 else step_content["context"]
                    st.write(context_preview)
                    print(f"Context for {step_name} (first 100 characters): {step_content['context'][:100]}...")
                else:
                    print(f"No context found for step: {step_name}")
                
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

            if "context" in final_state:
                st.subheader("Final Context Used")
                context_preview = final_state["context"][:500] + "..." if len(final_state["context"]) > 500 else final_state["context"]
                st.write(context_preview)
                print(f"Final context (first 100 characters): {final_state['context'][:100]}...")
            else:
                print("No final context found")


if __name__ == "__main__":
    main()