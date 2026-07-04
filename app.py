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

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import streamlit as st

# Configure the Streamlit app
st.set_page_config(page_title="AI Blog Search - Agentic RAG", page_icon=":mag_right:", layout="centered")

# Inject premium CSS styling
st.markdown("""
<style>
/* Import Outfit font */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Outfit', sans-serif;
}

/* Custom titles and headers */
h1, h2, h3 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
}

/* Styling sidebar cards */
section[data-testid="stSidebar"] {
    border-right: 1px solid #e2e8f0;
}

/* Styling tabs in Streamlit */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0px 0px;
    padding: 8px 16px;
    font-weight: 500;
}

/* Custom premium button design */
div.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

st.header("⚡ :blue[Agentic RAG] :grey[AI Blog & Document Search]")
st.caption("Load URLs or upload files to start a conversational search using LangGraph agents.")

# Initialize session state variables if they don't exist
if 'qdrant_host' not in st.session_state:
    st.session_state.qdrant_host = os.getenv("QDRANT_HOST", ":memory:")
if 'qdrant_api_key' not in st.session_state:
    st.session_state.qdrant_api_key = os.getenv("QDRANT_API_KEY", "dummy")
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
if 'messages' not in st.session_state:
    st.session_state.messages = []

def set_sidebar():
    """Setup sidebar for API keys and configuration."""
    with st.sidebar:
        st.subheader("⚙️ API Configuration")
        
        qdrant_host = st.text_input("Qdrant Host URL:", type="password", value=st.session_state.qdrant_host)
        qdrant_api_key = st.text_input("Qdrant API Key:", type="password", value=st.session_state.qdrant_api_key)
        gemini_api_key = st.text_input("Gemini API Key:", type="password", value=st.session_state.gemini_api_key)

        if st.button("Save Configuration", use_container_width=True):
            if qdrant_host and qdrant_api_key and gemini_api_key:
                st.session_state.qdrant_host = qdrant_host
                st.session_state.qdrant_api_key = qdrant_api_key
                st.session_state.gemini_api_key = gemini_api_key
                st.success("API keys saved successfully!")
                st.rerun()
            else:
                st.warning("Please fill all configuration fields.")
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat history cleared!")
            st.rerun()

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
            from google import genai
            client = genai.Client(api_key=st.session_state.gemini_api_key)
            available = [m.name for m in client.models.list() if 'embedContent' in getattr(m, 'supported_actions', [])]
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
        st.info("👋 Welcome! Please configure your Gemini API Key below to get started.")
        with st.form("api_config_form"):
            gemini_key = st.text_input("Gemini API Key:", type="password", value=st.session_state.gemini_api_key, placeholder="Starts with AIzaSy...")
            
            st.markdown("**Qdrant Settings (Advanced)**")
            col1, col2 = st.columns(2)
            with col1:
                q_host = st.text_input("Qdrant Host URL:", value=st.session_state.qdrant_host or ":memory:")
            with col2:
                q_key = st.text_input("Qdrant API Key:", type="password", value=st.session_state.qdrant_api_key or "dummy")
                
            submitted = st.form_submit_button("Start Application", use_container_width=True)
            if submitted:
                if gemini_key:
                    st.session_state.gemini_api_key = gemini_key
                    st.session_state.qdrant_host = q_host
                    st.session_state.qdrant_api_key = q_key
                    st.success("Configuration saved! Initializing application...")
                    st.rerun()
                else:
                    st.error("Please enter a Gemini API Key to proceed.")
        return

    # Initialize components
    embedding_model, client, db = initialize_components()
    if not all([embedding_model, client, db]):
        return

    # Initialize retriever and tools
    retriever_tool = create_rag_retriever_tool(db)

    # Ingestion Selector UI
    st.markdown("### 📥 Document Ingestion")
    tab1, tab2 = st.tabs([":link: Query Blog URL", ":file_folder: Upload Local File (PDF/TXT)"])

    with tab1:
        url = st.text_input(
            "Enter web page link:",
            placeholder="e.g., https://lilianweng.github.io/posts/2023-06-23-agent/",
            key="url_input"
        )
        if st.button("Index Web Page", use_container_width=True):
            if url:
                with st.spinner("Scraping and indexing web page..."):
                    if add_documents_to_qdrant(url, db):
                        st.success("Web page indexed and saved to database successfully!")
                    else:
                        st.error("Failed to index web page.")
            else:
                st.warning("Please enter a URL first.")

    with tab2:
        uploaded_file = st.file_uploader("Upload a file:", type=["pdf", "txt"], key="file_upload")
        if st.button("Index Local File", use_container_width=True):
            if uploaded_file:
                with st.spinner(f"Parsing and indexing '{uploaded_file.name}'..."):
                    if add_uploaded_file_to_qdrant(uploaded_file, db):
                        st.success(f"File '{uploaded_file.name}' indexed and saved to database successfully!")
                    else:
                        st.error("Failed to process local file.")
            else:
                st.warning("Please choose a file to upload first.")

    st.markdown("---")
    st.markdown("### 💬 Chat with your Knowledge Base")

    # Render conversational chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Load agent graph
    graph = get_graph(retriever_tool)

    # Conversational chat input
    if prompt := st.chat_input("Ask a question about the indexed content..."):
        # Display user message in chat container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Store user message in history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate agent reply
        inputs = {"messages": [HumanMessage(content=prompt)]}
        with st.chat_message("assistant"):
            with st.spinner("Consulting agent workflow..."):
                try:
                    response = generate_message(graph, inputs)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()