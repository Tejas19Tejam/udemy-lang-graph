from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from .react import llm, tools

SYSYEM_MESSAGE = "You are a helpful assistant that can use tools to answer questions."


# Create a Reasoning Node
# Give the current conversation to the LLM and add its response to the graph state.
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke(
        [{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]]
    )
    return {"messages": [response]}


# Execute the tool calls requested by the model and add their results to the state.
# ToolNode is responsible to execute the tool and it store their result to the state
tool_node = ToolNode(tools)
