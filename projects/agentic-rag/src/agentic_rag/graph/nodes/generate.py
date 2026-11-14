from typing import Any, Dict

from agentic_rag.graph.chains.generation import generation_chain
from agentic_rag.graph.state import GraphState


def generate_node(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
