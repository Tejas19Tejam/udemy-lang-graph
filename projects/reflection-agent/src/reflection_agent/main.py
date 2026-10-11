from typing import TypedDict, Annotated

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, add_messages, END
from reflection_agent.chains import generate_chain, reflect_chain

"""
Here, you are telling LangGraph exactly how your graph's state should behave.
- MessageGraph: Defines the structure of your state.
- messages: The field that stores messages.
- list[BaseMessage]: The field contains a list of messages.
- Annotated[..., add_messages]: Attaches the add_messages reducer, which LangGraph uses to merge message updates according to its message-handling rules.

"""


# Create a TypedDict to define the structure of the state graph
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Define the nodes name in the state graph
REFLECT = "reflect"
GENERATE = "generate"


# Define the nodes in the state graph
def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


def should_continue(state: MessageGraph) -> str:
    if len(state["messages"]) > 3:
        return END
    return REFLECT


builder = StateGraph(MessageGraph)
builder.add_node(GENERATE, generation_node).add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


# Define the execution flow of the state graph
builder.add_conditional_edges(GENERATE, should_continue, {END: END, REFLECT: REFLECT})
builder.add_edge(REFLECT, GENERATE)


graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="reflection_graph.png")


if __name__ == "__main__":
    print("Hello from Reflection Agent Project !")

    res = graph.invoke({"messages": HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  " """)})
    print(res["messages"][-1].content)
