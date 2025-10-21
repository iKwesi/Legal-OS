"""Quick test of create_react_agent approach."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))
load_dotenv(backend_path / ".env")

from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

print("Testing create_react_agent...")

# Simple test tool
def test_tool(query: str) -> str:
    return f"Tool called with: {query}"

tools = [
    Tool(
        name="test_tool",
        description="A simple test tool",
        func=test_tool
    )
]

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Create agent with correct API
print("Creating agent...")
try:
    # Try new API (LangGraph v1.0+)
    agent = create_react_agent(
        model=llm,
        tools=tools,
    )
    print("Agent created successfully with new API!")
except TypeError as e:
    print(f"New API failed: {e}")
    # Try old API
    from langchain.agents import create_agent
    agent = create_agent(
        model=llm,
        tools=tools,
    )
    print("Agent created successfully with old API!")

# Test execution
print("\nTesting agent execution...")
try:
    result = agent.invoke({
        "messages": [HumanMessage(content="Use the test tool with input 'hello'")]
    })
    print(f"✅ Agent executed successfully!")
    print(f"Messages: {len(result.get('messages', []))}")
    for i, msg in enumerate(result.get('messages', [])[:5]):
        print(f"  Message {i}: {type(msg).__name__}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
