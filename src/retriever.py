from langchain_core.tools.retriever import create_retriever_tool

def create_rag_retriever_tool(db, name: str = "retrieve_documents", description: str = None):
    """Creates a retriever tool that interfaces the vector database with the LLM Agent."""
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    if description is None:
        description = (
            "Search and return information from the indexed documents. "
            "Use this tool to find information related to the user's question from the uploaded files or URLs."
        )
    return create_retriever_tool(retriever, name, description)
