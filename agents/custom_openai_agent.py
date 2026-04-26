from crewai import LLM
import os

llm = LLM(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        temperature=0.5,
    )

def generate_response(prompt: str) -> str:
    response = llm.call(prompt)
    return str(response)