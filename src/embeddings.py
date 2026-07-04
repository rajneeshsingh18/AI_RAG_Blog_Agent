from src.llm_provider import get_embedding_model as factory_get_embedding_model
import streamlit as st

def get_embedding_model(api_key: str = None):
    """Initializes the embedding model based on the selected provider in session state."""
    provider = st.session_state.get("provider", "ollama")
    return factory_get_embedding_model(provider, api_key=api_key)
