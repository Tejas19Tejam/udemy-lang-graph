from typing import Any, Dict

from agentic_rag.graph.state import GraphState
from ingestion import retriever

# Extract relevant documents from the Vector Store based of the question
def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    print(f"Retrieved {len(documents)} documents")
    return {"documents": documents, "question": question}