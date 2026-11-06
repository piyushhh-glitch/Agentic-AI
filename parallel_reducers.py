# Parallel Workflow 

import langchain
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END 
from typing import TypedDict,Annotated
from dotenv import load_dotenv

load_dotenv()

llm=ChatGroq(
    model='openai/gpt-oss-20b',
    temperature=0.1
)

def merge_score_dict(existing:dict,newupdate:dict) -> dict:
    if not existing:
        return newupdate
    return {**existing,**newupdate}

# State
class AnalyzerState(TypedDict):
    raw_text : str
    safety_scores : Annotated[dict[str,int],merge_score_dict]


# Nodes

def toxicity_node(state:AnalyzerState) -> dict:
    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
        )

    response=llm.invoke(prompt)
    try:
        score=int(response.content.strip())
    except:
        score=0

    # Return a sub-dictionary under the EXACT SAME state key
    return {'safety_scores':{'toxicity_level':score}}

def copyright_node(state: AnalyzerState) -> dict:
    print("\n🔏 [Branch 2] Analyzing Copyright & Originality Risks...")
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_scores": {"copyright_risk": score}}


def culture_node(state: AnalyzerState) -> dict:
    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_scores": {"cultural_insensitivity": score}}

graph=StateGraph(AnalyzerState)

graph.add_node('toxicity',toxicity_node)
graph.add_node('copyright',copyright_node)
graph.add_node('culture',culture_node)

graph.add_edge(START,'toxicity')
graph.add_edge(START,'copyright')
graph.add_edge(START,'culture')

graph.add_edge('toxicity',END)
graph.add_edge('copyright',END)
graph.add_edge('culture',END)

app=graph.compile()

sample_script = """
    Yo guys! Welcome back to the stream. Today I am going to show you how to hack into 
    your friend's system using a script I copied directly from an online forum. 
    Honestly, traditional security protocols are absolute garbage and anyone still using 
    them is an absolute idiot. Let's dive into the code!
    """

initial_state={
    "raw_text":sample_script,
    "safety_scores":{}
}

final_state=app.invoke(initial_state)

print(final_state['safety_scores'])


# PARALLEL WORKFLOW

# 1. User provides raw text as input.
#
# 2. START triggers three parallel nodes:
#       - Toxicity Node
#       - Copyright Node
#       - Culture Node
#
# 3. Toxicity Node:
#       - Checks profanity, aggression and hate speech.
#       - Generates toxicity score.
#
# 4. Copyright Node:
#       - Checks plagiarism and originality risks.
#       - Generates copyright risk score.
#
# 5. Culture Node:
#       - Checks regional and cultural sensitivity.
#       - Generates cultural sensitivity score.
#
# 6. All three nodes update the SAME state key:
#       safety_scores
#
# 7. Custom Reducer:
#       - Receives existing and new dictionary updates.
#       - Merges all three scores into one dictionary.
#
# 8. LangGraph waits for all parallel branches to finish.
#
# 9. Final state contains all three safety scores.
#
# FLOW:
# START → [TOXICITY | COPYRIGHT | CULTURE]
#       → REDUCER → END
#
# ARCHITECTURE: Parallel Workflow