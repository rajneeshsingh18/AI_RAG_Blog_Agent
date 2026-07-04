from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs, chunk_size: int = 100, chunk_overlap: int = 50):
    """Splits loaded LangChain Documents into smaller, overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)
