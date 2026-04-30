from langchain_ollama import ChatOllama

# temperature=0 enforces deterministic, consistent reasoning across all agents.
_llm = ChatOllama(model="llama3.2", temperature=0)
