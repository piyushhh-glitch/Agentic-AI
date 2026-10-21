from crewai import Agent, Crew, Task, LLM
from crewai_tools import SerperDevTool
import os 
from dotenv import load_dotenv

load_dotenv()

llm=LLM(
    model="gemini/gemini-3.5-flash-lite",
    
    api_key=os.getenv("GEMINI_API_KEY")
)

search_tool=SerperDevTool()

research_agent=Agent(
    role="AI Research Analyst",
    goal="Research topics using web search and provide accurate , useful findings",
    backstory=(
        "You are a technical research analyst. "
        "You search for relevant information, compare findings, "
        "and explain results clearly."
    ),
    llm=llm,
    tools=[search_tool],
    verbose=True,
    max_iter=2,
    max_retry_limit=1
)

research_task = Task(
    description=(
        "Research {topic}. Use the web search tool to find relevant "
        "information. Summarize the key findings, explain important "
        "developments, and include source URLs when available. "
        "Do not invent facts or sources."
    ),
    expected_output=(
        "A concise research report with an overview, key findings, "
        "important developments, and source URLs."
    ),
    agent=research_agent
)

research_crew=Crew(
    agents=[research_agent],
    tasks=[research_task],
    verbose=True
)

response=research_crew.kickoff(
    inputs={"topic":"Latest development in Agentic AI"}
)

print("\nFINAL RESEARCH REPORT\n")
print(response)