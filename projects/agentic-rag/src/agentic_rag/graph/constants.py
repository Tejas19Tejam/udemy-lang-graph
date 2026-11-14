import os

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model_name: str = os.getenv("GROQ_LLM_MODEL_NAME") or ""

chat_llm = ChatGroq(model=model_name, temperature=0)


RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"
