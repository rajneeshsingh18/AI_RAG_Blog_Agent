from langchain_core.tools.retriever import create_retriever_tool

def create_rag_retriever_tool(db, name: str = "retrieve_blog_posts", description: str = None):
    """Creates a retriever tool that interfaces the vector database with the LLM Agent."""
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    if description is None:
        description = (
            "Search and return information about blog posts on LLMs, LLM agents, "
            "prompt engineering, and adversarial attacks on LLMs."
        )
    return create_retriever_tool(retriever, name, description)
