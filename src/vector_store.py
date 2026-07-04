from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from uuid import uuid4

def initialize_qdrant_client(host: str, api_key: str):
    """Initializes the Qdrant Client."""
    return QdrantClient(host, api_key=api_key)

def ensure_collection_exists(client: QdrantClient, collection_name: str, vector_size: int = 768):
    """Checks if the collection exists, and if not, creates it."""
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

def get_vector_store(client: QdrantClient, collection_name: str, embedding_model):
    """Returns the QdrantVectorStore instance."""
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model
    )

def add_documents_to_db(db: QdrantVectorStore, doc_chunks):
    """Adds document chunks with generated UUIDs to the Qdrant vector store."""
    uuids = [str(uuid4()) for _ in range(len(doc_chunks))]
    db.add_documents(documents=doc_chunks, ids=uuids)
