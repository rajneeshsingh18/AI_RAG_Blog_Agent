import os
import tempfile
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader

def load_web_page(url: str):
    """Loads HTML text from a URL and converts it into standard LangChain Document objects."""
    return WebBaseLoader(url).load()

def load_uploaded_file(uploaded_file):
    """Saves Streamlit UploadedFile to a temp file, parses it to Documents, and cleans up."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
        elif suffix == ".txt":
            loader = TextLoader(temp_path, encoding="utf-8")
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Add source file name to document metadata
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
            
        return docs
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
