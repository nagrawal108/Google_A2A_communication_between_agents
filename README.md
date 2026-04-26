# Multi-Agent Collaboration with LangGraph Using the A2A Protocol

## Overview

This project showcases **seamless multi-agent collaboration** across heterogeneous frameworks using the **Agent-to-Agent (A2A) Protocol** and **LangGraph**.

It orchestrates a pipeline in which:

- A **LangChain Agent** node extracts the task from the user's query.
- A **CrewAI Agent** conducts web research using **Tavily Search**.
- A **Response Agent** (CrewAI LLM) synthesizes the final answer.
- **LangGraph** connects all agents within a stateful, sequential execution graph.
- All LLM calls use **Groq** with the **Llama 3.3 70B** model.

---

## Problem Statement

### Business Need

Enterprises often rely on multiple AI agents optimized for specific functions—intent parsing, research, and content generation. These agents may be built with LangChain, CrewAI, or direct API calls. Coordinating them into a single, reliable pipeline is often complex, error-prone, and difficult to adapt.

### Objective

Design a flexible, modular system using the **A2A protocol** to:

- Enable agent interoperability,
- Maintain shared context,
- Support dynamic routing and collaboration among agents,
- Demonstrate plug-and-play capabilities across frameworks.

---

## Solution Approach

Use **LangGraph** to orchestrate the workflow:

1. **LangChain Agent Node**  
   Extracts the core task from user input.

2. **CrewAI Agent Node**  
   Executes real-time web research via Tavily Search, returns 5 bullet points.

3. **Response Agent Node**  
   Produces a synthesized response using Groq (Llama 3.3 70B) via CrewAI's LLM class.

4. **LangGraph**  
   Manages state transitions and context sharing in line with A2A principles.

---

## Architecture Diagram

```
[User Query]
     ↓
[LangChain Agent Node]  ← extracts task from input
     ↓ (task)
[CrewAI Agent Node]      ← web research via Tavily
     ↓ (web_results)
[Response Agent Node]    ← final answer via Groq/Llama 3.3 70B
     ↓ (final_response)
  [Output to User]
```

---

## Project Structure

```
├── main.py                      # LangGraph orchestrator — builds and runs the agent pipeline
├── agents/
│   ├── langchain_agent.py       # LangChain agent definition (Groq + task extraction tool)
│   ├── crewai_agent.py          # CrewAI researcher agent (Groq + Tavily web search)
│   └── custom_openai_agent.py   # Response generation agent (CrewAI LLM via Groq)
├── .env                         # Environment variables (API keys)
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your-groq-api-key
TAVILY_API_KEY=your-tavily-api-key
```

- **GROQ_API_KEY** — Get a free key at [console.groq.com](https://console.groq.com)
- **TAVILY_API_KEY** — Get a key at [tavily.com](https://tavily.com)

---

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install "crewai[azure-ai-inference]"
```

---

## Usage

```bash
python main.py
```

This runs the full pipeline: task extraction → web research → final response generation.
