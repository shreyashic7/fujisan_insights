from dotenv import load_dotenv
load_dotenv()
import sys
import asyncio

# Fix Windows Proactor event loop traceback errors
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os


import streamlit as st
from retriever import get_retriever
from llm_chain import generate_rag_response

# Page Configuration
st.set_page_config(
    page_title="Japanese E-Commerce RAG Engine",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Multilingual RAG Engine for Japanese E-Commerce")
st.markdown("Query customer reviews seamlessly across languages powered by local **ChromaDB**, **Multilingual E5 Embeddings**, and **Google Gemini**.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ RAG Settings")
    api_key_input = st.text_input("Google AI Studio API Key", type="password", help="Enter your free Gemini API key.")
    k_val = st.slider("Retrieved Chunks (Top-K)", min_value=2, max_value=10, value=4)
    
    st.markdown("---")
    st.markdown("### Capstone Project Info")
    st.markdown("- **Dataset:** Amazon Reviews Multi (Japanese)")
    st.markdown("- **Vector Store:** Local ChromaDB")
    st.markdown("- **Embedding:** `multilingual-e5-base`")

# Set API key if provided via UI
if api_key_input:
    os.environ["GEMINI_API_KEY"] = api_key_input

# Initialize retriever session state
@st.cache_resource
def load_rag_retriever(k):
    return get_retriever(k=k)

try:
    retriever = load_rag_retriever(k_val)
except Exception as e:
    st.error(f"Failed to load local ChromaDB. Make sure 'chroma_db_jp_reviews' is in your project directory. Error: {e}")
    st.stop()

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if user_query := st.chat_input("Ask about products, reviews, quality, or feedback (e.g., 'What do customers say about battery life?')..."):
    if not os.environ.get("GEMINI_API_KEY"):
        st.warning("Please enter your Google AI Studio API Key in the sidebar to proceed.")
        st.stop()
        
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching Japanese reviews and generating response..."):
            try:
                # E5 query prefix requirement for optimal search alignment
                prefixed_query = f"query: {user_query}"
                retrieved_docs = retriever.invoke(prefixed_query)
                
                # Generate LLM response via Gemini
                answer = generate_rag_response(user_query, retrieved_docs)
                st.markdown(answer)
                
                # Display retrieved raw context in an expander for transparency & grading
                with st.expander("🔍 View Retrieved Japanese Review Chunks (Context Sources)"):
                    for idx, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Source {idx+1}** | Rating: ⭐ {doc.metadata.get('stars', 'N/A')} / 5")
                        st.text(doc.page_content)
                        st.markdown("---")
                
                # Save assistant response to session state
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")