from crewai import Agent , LLM 
from dotenv import load_dotenv
import os 

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")

llm=LLM(
    model="openai/openai/gpt-oss-20b",
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key,
    custom_openai=True
)

agent=Agent(
    role="AI Assistant",
    goal="Provice clear and accurate answer to user's question",
    backstory="You are a helpful AI assistant skilled at explaining technical concepts",
    llm=llm,
    verbose=True
)

response=agent.kickoff(
    "Explain what Machine Learning in simple terms"
)

print("/n Final Answer:")
print(response)