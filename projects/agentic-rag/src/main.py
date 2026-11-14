from dotenv import load_dotenv

load_dotenv()

from agentic_rag.graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "What is your capital of India?"}))
