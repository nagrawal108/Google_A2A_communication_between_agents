# main.py — LangGraph orchestrator
# Chains three agents in a sequential pipeline:
#   1. LangChain Agent node: extracts the task from user input
#   2. CrewAI Agent node: performs web research via Tavily search
#   3. Response Agent node: generates a final answer using Groq (Llama 3.3 70B)
# This example demonstrates how to use LangGraph to build a simple agent orchestration pipeline, passing shared state between nodes and integrating multiple agent types.

from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from agents.langchain_agent import create_langchain_agent
from agents.crewai_agent import create_crewai_agent
from agents.custom_openai_agent import generate_response


### Define State — shared context passed between agent nodes
class A2AState(TypedDict, total=False):
    input: str
    task: str
    web_results: str
    final_response: str


### Define Nodes
def langchain_node(state):
    # Node 1: Pass user input through as the task for downstream agents
    print("state is: ", state)
    return {"task": state["input"]}


def crewai_node(state):
    # Node 2: CrewAI researcher agent searches the web using Tavily
    print(" state task is: ", state["task"])
    crew = create_crewai_agent(task_description=state["task"])
    result = crew.kickoff()
    return {"web_results": str(result)}


def openai_response_node(state):
    # Node 3: Generate final response using Groq (Llama 3.3 70B) via CrewAI's LLM
    prompt = f"Based on the web results: {state['web_results']}, respond to the task: {state['task']}"
    response = generate_response(prompt)
    return {"final_response": response}


### Build Graph — sequential pipeline: LangChain → CrewAI → Response
builder = StateGraph(A2AState)
builder.add_node("LangChainAgent", langchain_node)
builder.add_node("CrewAIAgent", crewai_node)
builder.add_node("OpenAIAgent", openai_response_node)

# Edges
builder.set_entry_point("LangChainAgent")
builder.add_edge("LangChainAgent", "CrewAIAgent")
builder.add_edge("CrewAIAgent", "OpenAIAgent")
builder.add_edge("OpenAIAgent", END)

graph = builder.compile()

### Execute
if __name__ == "__main__":
    input_text = "Can you summarize the latest trends in GenAI for enterprise?"
    output = graph.invoke({"input": input_text})
    print("Final Output:\n", output["final_response"])