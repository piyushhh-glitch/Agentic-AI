from typing import TypedDict 

from langgraph.graph import StateGraph, START, END 
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm=ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

class State(TypedDict):
    message: str

def chatbot(state:State):
    response=llm.invoke(state['message'])
    return {"message":response.content}

builder=StateGraph(State)

builder.add_node("chatbot",chatbot)

builder.add_edge(START,"chatbot")
builder.add_edge("chatbot",END)

graph=builder.compile()

result=graph.invoke({"message": "What is Artificial Intelligent"})

print(result['message'])