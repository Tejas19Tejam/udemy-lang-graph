from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from reflexion_agent.models import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=2)


def run_queries(search_queries: list[str], **kwargs) -> list[dict]:
    """Run queries using the TavilySearch tool."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


# ToolNode that executes the TavilySearch tool for both AnswerQuestion and ReviseAnswer tasks.
execute_tool = ToolNode(
    [
        StructuredTool.from_function(
            func=run_queries,
            name=AnswerQuestion.__name__,
        ),
        StructuredTool.from_function(
            func=run_queries,
            name=ReviseAnswer.__name__,
        ),
    ]
)
