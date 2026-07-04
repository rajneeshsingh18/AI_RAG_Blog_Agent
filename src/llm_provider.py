from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

def get_chat_model(provider: str, api_key: str = None, model_name: str = None):
    """
    Returns the appropriate Chat model instance based on the provider.
    """
    if provider == "ollama":
        # Local Ollama model
        selected_model = model_name or "llama3.2"
        return ChatOllama(
            model=selected_model,
            temperature=0
        )
    else:
        # Gemini model
        selected_model = model_name or "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(
            api_key=api_key,
            model=selected_model,
            temperature=0,
            streaming=True
        )

def get_embedding_model(provider: str, api_key: str = None, model_name: str = None):
    """
    Returns the appropriate Embedding model instance based on the provider.
    """
    if provider == "ollama":
        # Local Ollama Embeddings
        selected_model = model_name or "nomic-embed-text"
        return OllamaEmbeddings(
            model=selected_model
        )
    else:
        # Gemini Embeddings
        selected_model = model_name or "models/gemini-embedding-001"
        return GoogleGenerativeAIEmbeddings(
            model=selected_model,
            google_api_key=api_key
        )

def get_vector_size(provider: str):
    """
    Returns the dimensions of the selected provider's default embedding model.
    """
    if provider == "ollama":
        # nomic-embed-text returns 768 dimensions
        return 768
    else:
        # gemini-embedding-001 returns 3072 dimensions
        return 3072
