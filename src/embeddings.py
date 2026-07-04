from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embedding_model(api_key: str):
    """Initializes Google Generative AI embedding model using the text-embedding-004 model."""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )
