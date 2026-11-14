from typing import NotRequired, TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: NotRequired[str]
    web_search: NotRequired[bool]
    documents: NotRequired[list[Document]]