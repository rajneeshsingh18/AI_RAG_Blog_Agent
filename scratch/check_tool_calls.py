from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def retrieve_documents(query: str) -> str:
    """Search and return information from the indexed documents."""
    return "Your secret code is 987654."

model = ChatOllama(model="qwen2.5-coder:7b", temperature=0)
model_with_tools = model.bind_tools([retrieve_documents])

res = model_with_tools.invoke([HumanMessage(content="What is my secret code?")])
print("AIMessage object:", res)
print("Type of res:", type(res))
print("Content:", res.content)
print("Tool calls:", res.tool_calls)
print("Additional kwargs:", res.additional_kwargs)
