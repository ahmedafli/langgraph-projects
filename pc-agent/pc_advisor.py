from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
import operator
import os
from dotenv import load_dotenv

load_dotenv()

# Brain
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Two search tools - one global, one local
global_search = TavilySearch(max_results=5)
local_search = TavilySearch(
    max_results=5,
    include_domains=["https://www.tunisianet.com.tn","https://megapc.tn/"]
)

# The shared memory of ALL agents
class State(TypedDict):
    messages: Annotated[list, operator.add]
    use_case: str
    budget: str
    user_location: str
    specs_research: str
    local_prices: str
    final_recommendation: str

# NODE 1 - Interview the user
def interviewer(state: State):
    print("\n🤖 PC Advisor starting...\n")
    use_case = input("What will you use the PC for? (gaming/programming/video editing/general): ")
    budget = input("What is your budget in TND?: ")
    location = input("What city are you in?: ")
    return {
        "use_case": use_case,
        "budget": budget,
        "user_location": location,
        "messages": [HumanMessage(content=f"I need a PC for {use_case}, budget {budget} TND, located in {location}")]
    }

# NODE 2 - Research best specs globally
def researcher(state: State):
    print("\n🔍 Researching best PC specs for your needs...")
    search_query = f"best PC build for {state['use_case']} {state['budget']} TND budget 2026 specs recommendations"
    results = global_search.invoke({"query": search_query})
    print(f"AAAAAAA222222222222222AAAAAAAAAAAAASearch query results found of researcher node 2 global search invoke query wich is best pc build for ...use case budget tnd budget 2026 specs recommendations: {(results)}")
    response = llm.invoke(f"""
    Based on these search results, extract the best PC components
    (CPU, GPU, RAM, Storage) for someone who needs a PC for {state['use_case']}
    with a budget of {state['budget']} TND.
    Be specific with component names and models.
    Format your response as:
    CPU: <recommendation>
    GPU: <recommendation>
    RAM: <recommendation>
    STORAGE: <recommendation>
    REASON: <why these components>
    Search results:
    {str(results)[:2000]}
    """)
    print(f"response of researcher node 2 global search invoke llm invoke extract the best pc components cpu gpu ram storage for someone who needs a pc for use case with a budget of budget tnd and we gave it the search results: {response}")
    return {
        "specs_research": response.content,
        "messages": [AIMessage(content=f"Research done: {response.content}")]
    }

# NODE 3 - Search local Tunisian prices
def price_researcher(state: State):
    print("\n💰 Searching local prices in Tunisia...")
    print(f"AAAAAAAAAAAAAAAAAAAAAAAAAAstate specs research in price researcher node 3: {state['specs_research']}")
    # Extract component names from specs research
    response = llm.invoke(f"""
    Extract ONLY the component model names from this text, nothing else.
    Format: CPU model, GPU model, RAM model, Storage model
    Text: {state['specs_research']}
    """)
    print(f"response for price researcher node 3 the response of llm invoke extract only the componenet model names from this text and we gave it state specs search: {response}")
    components = response.content
    
    # Search local prices for each component
    search_query = f"{components} price Tunisia TND buy online"
    results = local_search.invoke({"query": search_query})
    
    response = llm.invoke(f"""
    Based on these search results from Tunisian stores, extract the prices
    for these components: {components}
    
    Format your response as:
    COMPONENT: <name> | PRICE: <price in TND> | STORE: <store name> | LINK: <url if available>
    
    If price not found for a component, write: COMPONENT: <name> | PRICE: Not found
    
    Search results:
    {str(results)[:2000]}
    """)
    
    return {
        "local_prices": response.content,
        "messages": [AIMessage(content=f"Prices found: {response.content}")]
    }

# NODE 4 - Analyze and compare everything
def analyst(state: State):
    print("\n📊 Analyzing and comparing options...")
    
    response = llm.invoke(f"""
    You are a PC buying expert. Based on the research and prices below,
    give a complete analysis.
    
    User needs: {state['use_case']}
    Budget: {state['budget']} TND
    Location: {state['user_location']}
    
    Recommended specs:
    {state['specs_research']}
    
    Local prices found:
    {state['local_prices']}
    
    Analyze:
    1. Does the total price fit the budget?
    2. Are there cheaper alternatives for any component?
    3. What is the best value option?
    4. Any warnings or things to watch out for?
    """)
    
    return {
        "messages": [AIMessage(content=f"Analysis: {response.content}")]
    }

# NODE 5 - Final recommendation
def advisor(state: State):
    print("\n✅ Generating final recommendation...")
    
    # Get full conversation context
    full_context = "\n".join([m.content for m in state['messages']])
    
    response = llm.invoke(f"""
    You are a friendly PC buying advisor. Based on everything below,
    give a clear, simple final recommendation.
    
    User needs: PC for {state['use_case']}, budget {state['budget']} TND, in {state['user_location']}
    
    Full research and analysis:
    {full_context[:3000]}
    
    Give your final answer in this format:
    
    🖥️ BEST PC BUILD FOR YOU:
    
    CPU: <component> - <price> TND
    GPU: <component> - <price> TND
    RAM: <component> - <price> TND
    STORAGE: <component> - <price> TND
    
    💰 TOTAL ESTIMATED COST: <total> TND
    
    🛒 WHERE TO BUY: <best store>
    
    ⚡ WHY THIS BUILD: <simple explanation>
    
    ⚠️ WATCH OUT FOR: <any warnings>
    """)
    
    return {
        "final_recommendation": response.content,
        "messages": [AIMessage(content=response.content)]
    }

# Build the graph
graph = StateGraph(State)

graph.add_node("interviewer", interviewer)
graph.add_node("researcher", researcher)
graph.add_node("price_researcher", price_researcher)
graph.add_node("analyst", analyst)
graph.add_node("advisor", advisor)

graph.set_entry_point("interviewer")
graph.add_edge("interviewer", "researcher")
graph.add_edge("researcher", "price_researcher")
graph.add_edge("price_researcher", "analyst")
graph.add_edge("analyst", "advisor")
graph.add_edge("advisor", END)

agent = graph.compile()

# Run it
print("=" * 50)
print("🖥️  PC BUYING ADVISOR - Powered by AI Agents")
print("=" * 50)

result = agent.invoke({
    "messages": [],
    "use_case": "",
    "budget": "",
    "user_location": "",
    "specs_research": "",
    "local_prices": "",
    "final_recommendation": ""
})

print("\n" + "=" * 50)
print(result['final_recommendation'])
print("=" * 50)