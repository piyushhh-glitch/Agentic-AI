# Sequential Workflow 

from typing import TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END 

class pipelinestate(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str

load_dotenv()

llm=ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

def editor(state:pipelinestate) -> dict:
    """"Stage 1 : Cleans up grammar,removes typos and refines the tones"""

    prompt = (
    "You are an expert copyeditor. Clean up the following raw text. "
    "Fix any grammatical errors, spelling mistakes, and smooth out the transition flow "
    "while keeping the core message intact. Return only the edited text.\n\n"
    f"Text:\n{state['raw_input']}"
        )

    response=llm.invoke(prompt)

    return {"edited_text":response.content.strip()}

def scriptwriter(state:pipelinestate) -> dict:
    """Stage 2: Formats the clean text into an engaging video script style."""

    print("\n--- [Stage 2] Executing Scriptwriter Node ---")

    prompt = (
    "You are a charismatic YouTube content creator. Take this edited text and transform "
    "it into a highly engaging, punchy, conversational video script hook. Make it sound "
    "like a real person speaking passionately. Return only the script content.\n\n"
    f"Edited Text:\n{state['edited_text']}"
        )

    response=llm.invoke(prompt)

    return {"script_text":response.content.strip()}


def translator(state:pipelinestate) -> dict:
    """Stage 3: Translates the script into natural flowing Hinglish."""

    print("\n--- [Stage 3] Executing Hinglish Translator Node ---")

    prompt = (
    "You are an expert content localizer for the Indian market. Take the following script "
    "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence-by-sentence "
    "or repeat information. Alternating comfortably between Hindi and English phrases just like "
    "an intellectual tech educator would speak naturally on a live stream. Keep the energy high. "
    "Return only the final Hinglish text.\n\n"
    f"Script:\n{state['script_text']}"
        )

    response=llm.invoke(prompt)

    return {"final_output":response.content.strip()}

graph=StateGraph(pipelinestate)

# Nodes --->
graph.add_node("editor",editor)
graph.add_node("scriptwriter",scriptwriter)
graph.add_node("translator",translator)


# Edges --->

graph.add_edge(START,'editor')
graph.add_edge('editor','scriptwriter')
graph.add_edge('scriptwriter','translator')
graph.add_edge('translator',END)

# Compile the graph

app=graph.compile()

result=app.invoke(
    {
        "raw_input":"artificial intelligence is changing the world very fast. today almost every industry is using ai to improve their work. companies are using machine learning to understand customer behaviour, automate tasks and make better decisions. but ai is not just about chatgpt or generating images. it is becoming an important part of healthcare, education, finance and robotics. in the coming years people who understand how to use ai will have a big advantage in the job market."
    }
)

# Output
print(result['final_output'])


# ==========================================
# WORKFLOW EXECUTION
# ==========================================

# 1. User provides raw input text.
# 2. START triggers the Editor Node.
# 3. Editor Node:
#       - Takes raw_input from state.
#       - Corrects grammar, spelling and sentence flow.
#       - Returns edited_text.
#
# 4. Scriptwriter Node:
#       - Takes edited_text from state.
#       - Converts it into an engaging YouTube video script.
#       - Returns script_text.
#
# 5. Translator Node:
#       - Takes script_text from state.
#       - Converts it into natural Hinglish.
#       - Returns final_output.
#
# 6. END terminates the workflow.
#
# 7. app.invoke() executes the complete graph.
#
# 8. Finally, print(result['final_output'])
#
# FLOW:
# START → EDITOR → SCRIPTWRITER → TRANSLATOR → END
#
# ARCHITECTURE: Sequential Workflow
# ==========================================