"""
generator.py — Uses Google Gemini API to synthesize answers from RAG chunks.

Reads the retrieved chunks and passes them as context to gemini-2.5-flash.
If GEMINI_API_KEY is missing, gracefully falls back to returning the raw chunk text.
"""

import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

def generate(query: str, chunks: list) -> str:
    """
    Synthesize an answer using the Gemini API based purely on retrieved chunks.
    No local extraction or regex matching is performed.
    """
    if not chunks:
        return (
            "I could not find relevant information for your query. "
            "Please contact customer support for assistance."
        )

    # Use the environment variable as requested by the user
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[generator] WARNING: GEMINI_API_KEY is not set. Returning raw chunk.")
        return chunks[0]["text"]

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Combine the text of all chunks into a single context string
        context_text = "\n\n---\n\n".join(
            [f"Source Document: {c['source']}\n{c['text']}" for c in chunks]
        )
        
        prompt = f"""You are a helpful, professional, and concise customer service assistant for TRAVIS Bank.
You have been given the following retrieved policy documents from the bank's knowledge base to help answer a customer's query.

<context>
{context_text}
</context>

<user_query>
{query}
</user_query>

Instructions:
1. Answer the user's query clearly and concisely using ONLY the provided context.
2. If the context does not contain the answer, simply state that you do not have that information in your policies and advise them to contact customer support. Do not hallucinate or guess.
3. Structure your response in a readable format (e.g. use bullet points if appropriate).
4. Do not mention "Based on the provided context" or similar phrases. Speak directly to the customer as if you inherently know the policies.
5. If the query is just a generic greeting, respond politely and ask how you can help with their banking needs today.
"""

        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.2, # Low temperature for factual RAG responses
            ),
        )
        return response.text.strip()
    except Exception as e:
        print(f"[generator] Error calling Gemini API: {e}")
        # Graceful fallback to raw chunk text in case of network or API errors
        return chunks[0]["text"]