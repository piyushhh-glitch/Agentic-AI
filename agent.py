from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

llm=ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

class State(TypedDict):
    question: str
    answer: str

def math_agent(state:State):
    response=llm.invoke(
        [
            ("system","You are a helpful AI Assistant.Help in calculations in simple english language"),
            ("human",state['question'])
        ]
    )

    return {"answer":response.content}

builder=StateGraph(State)

builder.add_node("math_agent",math_agent)

builder.add_edge(START,"math_agent")
builder.add_edge("math_agent",END)

graph=builder.compile()

result=graph.invoke(
    {
        "question":"If a train travels 120 km in 2 hours, calculate its speed.",
        "answer":""
    }
)

print("Question:",result["question"])
print("Answer:",result["answer"])


