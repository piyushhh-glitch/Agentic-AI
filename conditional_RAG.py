import os 
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph , START, END 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv 
from langchain_community.vectorstores import FAISS

load_dotenv()

# 1. Building the RAG Pipeline
embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def build_retriever(pdf_path:str):
    loader=PyPDFLoader(pdf_path)
    document=loader.load()

    splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)

    chunks=splitter.split_documents(document)

    vectorstore=FAISS.from_documents(chunks,embedding)

    return vectorstore.as_retriever(search_kwargs={"k":4})

academic_retriever=build_retriever("academics_handbook.pdf")
fee_retriever=build_retriever("fee_structure.pdf")

llm=ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.4
)

# 2. State
class State(TypedDict):
    programme: str
    messages: Annotated[list,add_messages]
    query_type: str
    retrieved_context: str


# 3. Nodes

def classifier_node(state:State)->dict:
    """Look at the latest message and decide which path to take."""

    last_message=state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
        )

    response=llm.invoke(prompt)

    category=response.content.strip().lower()

    if 'academic' in category:
        category='academic'
    elif 'fee' in category:
        category='fee'
    else:
        category='general'

    return {'query_type':category}


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state["messages"][-1].content
    docs = acedemic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}

# 4. Router function

def route_query(state:State):
    if state["query_type"]=='academic':
        return "academic_rag"
    elif state['query_type']=='fee':
        return "fee_rag"
    else:
        return "general"

# 5. Graph

graph=StateGraph(State)

graph.add_node('classifier',classifier_node)
graph.add_node('academic_rag',academic_rag_node)
graph.add_node('fee_rag',fee_rag_node)
graph.add_node('general',general_node)
graph.add_node('response',response_node)


# Edges

graph.add_edge(START,'classifier')

graph.add_conditional_edges('classifier',route_query)


graph.add_edge('academic_rag','response')
graph.add_edge('fee_rag','response')
graph.add_edge('general','response')

graph.add_edge('response',END)

app=graph.compile()

# 6. Run the code 
print("welcome to the College assistant \n\n")

print("which programe are you in ")
print("1. BCA")
print("2. BBA")
print("3. B.com (H)")

choice = input("\nEnter 1, 2 or 3 ")

programme_map = {
    "1": "BCA",
    "2": "BBA",
    "3": "B.Com (H)"
}
student_programme = programme_map.get(choice, "BCA")

print(f"\nGreat! You're set as a {student_programme} student.")

while True:
    user_query = input("You:  ")

    if user_query.lower() in ["exit","quit"]:
        break
    
    result = app.invoke({
        "programme": student_programme,
        "messages": [("human",user_query)]
    })

    print(f"Assistant : {result['messages'][-1].content}")



# ==============================================
# CONDITIONAL RAG WORKFLOW - COLLEGE ASSISTANT
# ==============================================

# 1. Student selects their programme (BCA/BBA/B.Com).
#
# 2. Student enters a query.
#
# 3. START triggers the Classifier Node.
#
# 4. CLASSIFIER NODE:
#       - Reads the latest user message.
#       - Sends the query to the LLM.
#       - Classifies it into academic, fee or general.
#       - Updates query_type in the state.
#
# 5. ROUTER FUNCTION:
#       - Checks query_type.
#       - Decides which node should execute next.
#
# 6. CONDITIONAL ROUTING:
#       - Academic -> Academic RAG Node.
#       - Fee      -> Fee RAG Node.
#       - General  -> General Node.
#
# 7. ACADEMIC RAG NODE:
#       - Searches the academic handbook PDF.
#       - Retrieves relevant chunks using FAISS.
#       - Stores results in retrieved_context.
#
# 8. FEE RAG NODE:
#       - Searches the fee structure PDF.
#       - Retrieves relevant chunks using FAISS.
#       - Stores results in retrieved_context.
#
# 9. GENERAL NODE:
#       - Skips document retrieval.
#       - Marks NO_RETRIEVAL_NEEDED in the state.
#
# 10. RESPONSE NODE:
#       - Receives the user query.
#       - Receives programme information.
#       - Uses retrieved context or general LLM knowledge.
#       - Generates the final response.
#
# 11. add_messages reducer updates message history.
#
# 12. END terminates the workflow.
#
# FLOW:
#
# START
#   |
# CLASSIFIER
#   |
# ROUTER
#   |
#   |------ Academic RAG ------|
#   |                          |
#   |------ Fee RAG -----------|---- RESPONSE ---- END
#   |                          |
#   |------ General -----------|
#
# ARCHITECTURE: CONDITIONAL RAG WORKFLOW
# ==============================================