from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END, START
from react_agent_fun_calling.nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST_MSG_INDEX = -1


# ```mermaid
# flowchart TD
#     __start__ --> agent_reason
#     agent_reason -->|tool call| act
#     agent_reason -->|final answer| __end__
#     act --> agent_reason
# ```

# Adding nodes to the Graph and entry point
flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.add_node(ACT, tool_node)

# It tells - when execution starts, go to AGENT_REASON
# Its equivalent to : flow.add_edge(START, AGENT_REASON)
flow.set_entry_point(AGENT_REASON)


def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST_MSG_INDEX].tool_calls:
        return END
    return ACT


# Allow AGENT_REASON node to optionally route to one or more edges
flow.add_conditional_edges(AGENT_REASON, should_continue, {END: END, ACT: ACT})


flow.add_edge(ACT, AGENT_REASON)


app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


if __name__ == "__main__":

    print("Hello from React Agent Function Calling Project !")

    res = app.invoke(
        {
            "messages": HumanMessage(
                content="What is the temperature in Mumbai? List it and triple it"
            )
        }
    )
    print(res["messages"][LAST_MSG_INDEX].content)
