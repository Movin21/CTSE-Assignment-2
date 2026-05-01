import os

from langchain_ollama import ChatOllama

# OLLAMA_HOST env var allows Docker to point to the ollama service container.
# Falls back to localhost for local development.
_llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.2"),
    base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    temperature=0,
)
