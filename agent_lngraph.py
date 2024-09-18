import os

from dotenv import load_dotenv
from openai import OpenAI
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()
openi_key= os.getenv("OPENAI_API_KEY")

llm_name= "gpt-4o-mini"
client= OpenAI(api_key= openi_key)


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]



graph_builder = StateGraph(State)

