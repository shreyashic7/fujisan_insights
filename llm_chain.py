import os
from groq import Groq

def generate_rag_response(query: str, retrieved_docs: list) -> str:
    """
    Sends the user query and retrieved Japanese review chunks to Groq's API,
    synthesizing an answer in the user's preferred language.
    """
    # Automatically retrieves GROQ_API_KEY from os.environ (.env)
    client = Groq()
    
    # Format retrieved context chunks
    context_text = "\n\n".join([
        f"[Source {i+1} | Rating: {doc.metadata.get('stars', 'N/A')} Stars]:\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    ])
    
    system_instruction = (
        "You are an expert AI assistant for a Japanese E-Commerce platform analysis tool. "
        "Your task is to analyze the provided Japanese customer reviews and answer the user's question. "
        "CRITICAL LANGUAGE RULE: Always reply in English by default, regardless of the language of the source reviews, "
        "UNLESS the user's query is explicitly written in Japanese or explicitly asks for an example in Japanese. "
        "Always cite the source review ratings and details where appropriate. If the context does not contain "
        "enough information, state clearly that there are no matching reviews found."
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {
            "role": "user", 
            "content": f"Context from Japanese E-Commerce Reviews:\n{context_text}\n\nUser Query: {query}\n\nAnswer:"
        }
    ]
    
    # Active open-weights model on Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.3,
    )
    
    return response.choices[0].message.content