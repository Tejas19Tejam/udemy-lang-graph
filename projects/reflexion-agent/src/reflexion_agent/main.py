from typing import Literal
from langgraph.graph import END, START, StateGraph, MessagesState
import reflexion_agent
from reflexion_agent.chains import first_responder, revisor
from reflexion_agent.tool_executor import execute_tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def draft_node(state: MessagesState):
    """
    Draft the initial response.
    """
    response = first_responder.invoke({"messages": state["messages"]})
    print(f"[Info] Draft node response : {response}")
    return {"messages": [response]}


def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""
    response = revisor.invoke({"messages": state["messages"]})

    print(f"[Info] Revise node response : {response}")

    return {"messages": [response]}


MAX_ITERATIONS = 2


def event_loop(state: MessagesState) -> Literal["executor", "end"]:
    """
    Event loop to handle the revision process.
    """

    iteration = sum(isinstance(item, ToolMessage) for item in state["messages"])

    if iteration < MAX_ITERATIONS:
        print(f"[Info] Iteration {iteration + 1} of {MAX_ITERATIONS}")
        return EXECUTOR
    else:
        print("[Info] Maximum iterations reached. Ending process.")
        return END


# Define the state graph
DRAFT = "draft"
REVISE = "revise"
EXECUTOR = "executor"

builder = StateGraph(MessagesState)

builder.add_node(DRAFT, draft_node)
builder.add_node(REVISE, revise_node)
builder.add_node(EXECUTOR, execute_tool)


builder.add_edge(START, DRAFT)
builder.add_edge(DRAFT, EXECUTOR)
builder.add_edge(EXECUTOR, REVISE)
builder.add_conditional_edges(REVISE, event_loop, {EXECUTOR: EXECUTOR, END: END})

graph = builder.compile()
graph.name = "Reflexion Agent Graph"

graph.get_graph().draw_mermaid_png(output_file_path="reflexion_graph.png")

if __name__ == "__main__":
    print("Reflexion Agent is running...")
    res = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital."
                )
            ]
        }
    )

    # Extract the final answer from the last message with tool calls
    last_message = res["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(last_message.tool_calls[0]["args"]["answer"])
    print(res)
