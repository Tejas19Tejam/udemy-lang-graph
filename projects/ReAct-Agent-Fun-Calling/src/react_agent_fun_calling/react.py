from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq

load_dotenv()


@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3


tools = [TavilySearch(max_results=1), triple]

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0).bind_tools(tools)
