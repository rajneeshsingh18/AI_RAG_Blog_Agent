import json
from langchain_core.messages import AIMessage
from langgraph.prebuilt import tools_condition

# Test parsing and updating AIMessage
content = '{"name": "retrieve_documents", "arguments": {"query": "secret code"}}'

try:
    parsed = json.loads(content)
    if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
        print("Successfully parsed as tool call!")
        
        # Format required for LangChain tool calls:
        # id is required, args is required (not arguments)
        tool_call = {
            "name": parsed["name"],
            "args": parsed["arguments"],
            "id": "call_test_123"
        }
        
        # Try creating AIMessage with tool_calls
        msg = AIMessage(content="", tool_calls=[tool_call])
        print("Created AIMessage:", msg)
        print("tool_calls attribute:", msg.tool_calls)
        
        # Test tools_condition
        state = {"messages": [msg]}
        res = tools_condition(state)
        print("tools_condition result:", res)
        
except Exception as e:
    print("FAILED:", e)
