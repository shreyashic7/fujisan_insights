import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "./chroma_db_jp_reviews"
COLLECTION_NAME = "japanese_amazon_reviews"

def get_retriever(k=4):
    """Initializes and returns the ChromaDB retriever using multilingual-e5-base."""
    # E5 models require query prefix for best cross-lingual retrieval
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Initialize Chroma vector store from local persistent folder
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    return vector_store.as_retriever(search_kwargs={"k": k})