from src.embeddings import get_embedding_model
from src.vector_store import (
    initialize_qdrant_client,
    ensure_collection_exists,
    get_vector_store,
    add_documents_to_db
)
from src.ingestion import load_web_page, load_uploaded_file
from src.chunking import chunk_documents
from src.retriever import create_rag_retriever_tool

from langchain_core.messages import HumanMessage
from src.graph import get_graph, generate_message

import streamlit as st

st.set_page_config(page_title="AI Blog Search", page_icon=":mag_right:")
st.header(":blue[Agentic RAG with LangGraph:] :green[AI Blog Search]")

# Initialize session state variables if they don't exist
if 'qdrant_host' not in st.session_state:
    st.session_state.qdrant_host = ""
if 'qdrant_api_key' not in st.session_state:
    st.session_state.qdrant_api_key = ""
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

def set_sidebar():
    """Setup sidebar for API keys and configuration."""
    with st.sidebar:
        st.subheader("API Configuration")
        
        qdrant_host = st.text_input("Enter your Qdrant Host URL:", type="password")
        qdrant_api_key = st.text_input("Enter your Qdrant API key:", type="password")
        gemini_api_key = st.text_input("Enter your Gemini API key:", type="password")

        if st.button("Done"):
            if qdrant_host and qdrant_api_key and gemini_api_key:
                st.session_state.qdrant_host = qdrant_host
                st.session_state.qdrant_api_key = qdrant_api_key
                st.session_state.gemini_api_key = gemini_api_key
                st.success("API keys saved!")
            else:
                st.warning("Please fill all API fields")

def initialize_components():
    """Initialize components that require API keys"""
    if not all([st.session_state.qdrant_host, 
               st.session_state.qdrant_api_key, 
               st.session_state.gemini_api_key]):
        return None, None, None

    try:
        embedding_model = get_embedding_model(st.session_state.gemini_api_key)

        # Initialize Qdrant client
        client = initialize_qdrant_client(
            st.session_state.qdrant_host,
            api_key=st.session_state.qdrant_api_key
        )

        # Ensure collection exists
        ensure_collection_exists(client, collection_name="qdrant_db")

        # Initialize vector store
        db = get_vector_store(client, collection_name="qdrant_db", embedding_model=embedding_model)

        return embedding_model, client, db
        
    except Exception as e:
        # Try listing models to diagnose
        try:
            # pyrefly: ignore [missing-import]
            import google.generativeai as genai
            genai.configure(api_key=st.session_state.gemini_api_key)
            available = [m.name for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
            st.error(f"Initialization error: {str(e)}")
            if available:
                st.info(f"Available embedding models for your API key: {available}")
            else:
                st.warning("No embedding models found with 'embedContent' support for this API key.")
        except Exception as list_err:
            st.error(f"Initialization error: {str(e)}\n\nDiagnostic error listing models: {str(list_err)}")
        return None, None, None



def add_documents_to_qdrant(url, db):
    try:
        docs = load_web_page(url)
        doc_chunks = chunk_documents(docs)
        add_documents_to_db(db, doc_chunks)
        return True
    except Exception as e:
        st.error(f"Error adding documents: {str(e)}")
        return False

def add_uploaded_file_to_qdrant(uploaded_file, db):
    try:
        docs = load_uploaded_file(uploaded_file)
        doc_chunks = chunk_documents(docs)
        add_documents_to_db(db, doc_chunks)
        return True
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        return False

def main():
    set_sidebar()

    # Check if API keys are set
    if not all([st.session_state.qdrant_host, 
                st.session_state.qdrant_api_key, 
                st.session_state.gemini_api_key]):
        st.warning("Please configure your API keys in the sidebar first")
        return

    # Initialize components
    embedding_model, client, db = initialize_components()
    if not all([embedding_model, client, db]):
        return

    # Initialize retriever and tools
    retriever_tool = create_rag_retriever_tool(db)
    tools = [retriever_tool]

    # Ingestion Selection
    tab1, tab2 = st.tabs([":link: Query Blog URL", ":file_folder: Upload Local File (PDF/TXT)"])

    with tab1:
        url = st.text_input(
            "Paste the blog link:",
            placeholder="e.g., https://lilianweng.github.io/posts/2023-06-23-agent/"
        )
        if st.button("Enter URL"):
            if url:
                with st.spinner("Processing documents..."):
                    if add_documents_to_qdrant(url, db):
                        st.success("Documents added successfully!")
                    else:
                        st.error("Failed to add documents")
            else:
                st.warning("Please enter a URL")

    with tab2:
        uploaded_file = st.file_uploader("Choose a PDF or TXT file:", type=["pdf", "txt"])
        if st.button("Enter File"):
            if uploaded_file:
                with st.spinner("Processing uploaded file..."):
                    if add_uploaded_file_to_qdrant(uploaded_file, db):
                        st.success(f"File '{uploaded_file.name}' processed successfully!")
                    else:
                        st.error("Failed to process file")
            else:
                st.warning("Please upload a file first")

    # Query section
    graph = get_graph(retriever_tool)
    query = st.text_area(
        ":bulb: Enter your query about the blog post:",
        placeholder="e.g., What does Lilian Weng say about the types of agent memory?"
    )

    if st.button("Submit Query"):
        if not query:
            st.warning("Please enter a query")
            return

        inputs = {"messages": [HumanMessage(content=query)]}
        with st.spinner("Generating response..."):
            try:
                response = generate_message(graph, inputs)
                st.write(response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    st.markdown("---")
    st.write("Built with :blue-background[LangChain] | :blue-background[LangGraph] by [Charan](https://www.linkedin.com/in/codewithcharan/)")

if __name__ == "__main__":
    main()