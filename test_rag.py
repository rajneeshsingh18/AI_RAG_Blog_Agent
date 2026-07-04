import sys
import os

# Mock streamlit session state before imports that might reference it
class MockSessionState(dict):
    def __getattr__(self, key):
        return self.get(key)
    def __setattr__(self, key, value):
        self[key] = value

import streamlit as st
st.session_state = MockSessionState({
    "provider": "ollama",
    "gemini_api_key": None,
    "qdrant_host": "http://localhost:6333",
    "qdrant_api_key": None
})

from src.llm_provider import get_chat_model, get_embedding_model, get_vector_size
from src.vector_store import initialize_qdrant_client, ensure_collection_exists, get_vector_store, add_documents_to_db
from src.retriever import create_rag_retriever_tool
from src.graph import get_graph, generate_message
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

print("=== STEP 1: INITIALIZING OLLAMA EMBEDDING MODEL ===")
try:
    embeddings = get_embedding_model("ollama")
    print("Success: Embedding model initialized.")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n=== STEP 2: CONNECTING TO QDRANT & ENSURING COLLECTION ===")
try:
    # Use in-memory Qdrant client to make the test self-contained and run-independent
    from qdrant_client import QdrantClient
    client = QdrantClient(":memory:")
    vector_size = get_vector_size("ollama")
    ensure_collection_exists(client, "test_collection", vector_size=vector_size)
    db = get_vector_store(client, "test_collection", embeddings)
    print("Success: Connected to in-memory Qdrant and created 'test_collection'.")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n=== STEP 3: INDEXING SAMPLE DOCUMENT ===")
try:
    test_doc = Document(page_content="Your secret code is 987654. Keep it safe!", metadata={"source": "test.txt"})
    add_documents_to_db(db, [test_doc])
    print("Success: Indexed 1 document.")
    # Check count
    count = client.count(collection_name="test_collection").count
    print(f"Documents in collection count: {count}")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n=== STEP 4: RETRIEVAL TEST ===")
try:
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    docs = retriever.invoke("What is my secret code?")
    print("Retrieved docs:")
    for d in docs:
        print(f"- Content: {d.page_content} (Source: {d.metadata.get('source')})")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n=== STEP 5: INITIALIZING CHAT MODEL ===")
try:
    chat_model = get_chat_model("ollama")
    print(f"Success: Chat model initialized: {chat_model.model}")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)

print("\n=== STEP 6: TEST LANGGRAPH WORKFLOW ===")
try:
    retriever_tool = create_rag_retriever_tool(db)
    # Mock session_state.db in streamlit session state because graph.py might read it
    st.session_state.db = db
    graph = get_graph(retriever_tool)
    
    question = "What is my secret code?"
    print(f"Asking graph: '{question}'")
    inputs = {"messages": [HumanMessage(content=question)]}
    
    # We will step through the graph stream to print the nodes executing!
    for output in graph.stream(inputs):
        for key, value in output.items():
            print(f"\n-> Node Executed: '{key}'")
            if isinstance(value, dict) and "messages" in value:
                msg = value["messages"][-1]
                print(f"   Message Type: {type(msg).__name__}")
                content_preview = msg.content if hasattr(msg, "content") else str(msg)
                if len(content_preview) > 100:
                    content_preview = content_preview[:100] + "..."
                print(f"   Content: {content_preview}")
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"   Tool Calls: {msg.tool_calls}")

    # Generate final answer using our generate_message function
    response = generate_message(graph, {"messages": [HumanMessage(content=question)]})
    print("\n==================================")
    print(f"FINAL RESPONSE:\n{response}")
    print("==================================")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
