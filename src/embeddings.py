from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embedding_model(api_key: str):
    """Initializes Google Generative AI embedding model using the gemini-embedding-001 model."""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
