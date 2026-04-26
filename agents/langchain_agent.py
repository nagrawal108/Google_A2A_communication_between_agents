from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import os


@tool
def extract_task_from_input(input_text: str) -> str:
    """Extract the core task from user input."""
    return f"Extracted task from input: {input_text.strip()}"


def create_langchain_agent():
    # Llama 3.3 70B on Groq — fastest free option with full tool calling support
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Add it to your .env file.")
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.5,
    )
    tools = [extract_task_from_input]

    agent = create_agent(
        model=llm,
        tools=tools,
    )
    return agent