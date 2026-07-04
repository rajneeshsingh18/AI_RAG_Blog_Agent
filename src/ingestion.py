from langchain_community.document_loaders import WebBaseLoader

def load_web_page(url: str):
    """Loads HTML text from a URL and converts it into standard LangChain Document objects."""
    return WebBaseLoader(url).load()
