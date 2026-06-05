from langgraph.graph import StateGraph, END
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, ToolMessage
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

# The brain
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm = ChatGroq(model="llama-3.3-70b-versatile")

# The search tool
search_tool = TavilySearch(max_results=2)

# Give the brain access to the tool
llm_with_tools = llm.bind_tools([search_tool])

class State(TypedDict):
    messages: list

def think(state: State):
    response = llm_with_tools.invoke(state["messages"])
    #print(f"LLM response: {response}")
    x = {"messages": state["messages"] + [response]}
    #print("hbhb",x)
    return {"messages": state["messages"] + [response]}

def use_tool(state: State):
    last_message = state["messages"][-1]
    tool_results = []
    for tool_call in last_message.tool_calls:
        result = search_tool.invoke(tool_call["args"])
        #print("result of the args shit tool call",result)
        tool_results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )
    return {"messages": state["messages"] + tool_results}

def should_use_tool(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "use_tool"
    return END

# Build the graph
graph = StateGraph(State)
graph.add_node("think", think)
graph.add_node("use_tool", use_tool)
graph.set_entry_point("think")
graph.add_conditional_edges("think", should_use_tool)
graph.add_edge("use_tool", "think")
agent = graph.compile()

# Chat
conversation = []
while True:
    user_input = input("You: ")
    conversation.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation})
    conversation = result["messages"]
    print(f"Agent: {conversation[-1].content}\n")