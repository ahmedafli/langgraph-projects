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

# Search tool
global_search = TavilySearch(max_results=5)

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

def researcher(state: State):
    print("\n🔍 Researching best PC specs for your needs...")
    
    is_laptop = any(word in state['use_case'].lower() for word in ['laptop', 'portable', 'notebook'])
    
    if is_laptop:
        search_query = f"meilleur laptop gamer tunisie {state['budget']} TND disponible acheter 2026"
    else:
        search_query = f"meilleur PC gamer tunisie {state['budget']} TND composants disponible 2026"
    
    results = global_search.invoke({"query": search_query})
    
    response = llm.invoke(f"""
    Based on these search results from Tunisian stores, recommend 3 specific laptop models
    that are actually available to buy in Tunisia within a budget of {state['budget']} TND.
    
    STRICT RULES:
    - Only recommend laptops that appear in the search results from Tunisian stores
    - Budget is {state['budget']} TND — don't recommend anything above this
    - Prefer brands commonly found in Tunisia: ASUS, MSI, Lenovo, HP, Dell, Acer
    - No Razer, no Alienware — too rare in Tunisia
    
    Format:
    OPTION1: <exact laptop model as it appears in Tunisian stores>
    OPTION2: <exact laptop model>
    OPTION3: <exact laptop model>
    REASON: <why these for gaming and this budget>
    
    Search results:
    {str(results)[:3000]}
    """)
    
    return {
        "specs_research": response.content,
        "messages": [AIMessage(content=f"Research done: {response.content}")]
    }

def price_researcher(state: State):
    print("\n💰 Searching local prices in Tunisia...")

    # Extract ALL recommended models
    response = llm.invoke(f"""
    From this text, extract all laptop/product model names mentioned.
    Return each on a new line, nothing else.
    
    Text: {state['specs_research']}
    """)
    
    models = [line.strip() for line in response.content.strip().split("\n") 
              if line.strip() and not line.startswith("REASON")]
    print(f"🎯 Searching for: {models}")

    tunisian_stores = [
        "mytek.tn",
        "spacenet.tn",
        "tunisianet.com.tn",
        "sbsinformatique.com",
        "wiki.tn",
        "megapc.tn",
        "expert-gaming.tn",
        "zstore.com.tn",
        "carthagoinformatique.tn",
        "bestbuytunisie.tn",
        "tunewtec.com",
        "scoop.com.tn",
    ]

    all_results = []

    # Search each model on each store
    for model in models[:3]:
        print(f"\n📦 Searching for: {model}")
        for store in tunisian_stores:
            query = f"{model} site:{store}"
            print(f"  🔎 {store}...")
            try:
                results = global_search.invoke({"query": query})
                for r in results["results"]:
                    if store.split(".")[0] in r["url"] and r["content"].strip():
                        all_results.append(
                            f"MODEL: {model}\nSTORE: {store}\nURL: {r['url']}\nDETAILS: {r['content'][:300]}"
                        )
                        break
            except:
                pass

    prices_text = "\n---\n".join(all_results) if all_results else "No local prices found"
    print(f"\n✅ Found {len(all_results)} results total")

    response = llm.invoke(f"""
    Extract prices from these Tunisian store results.
    
    RULES:
    - Ignore monthly installment prices (3 mois, 6 mois, 9 mois, 12 mois)
    - Only use full product prices
    - Note stock status
    - Include exact URLs
    
    Format each as:
    MODEL: <name>
    STORE: <store>
    PRICE: <full price> TND
    LINK: <url>
    STOCK: <status>
    ---
    
    Results:
    {prices_text}
    """)

    return {
        "local_prices": response.content,
        "messages": [AIMessage(content=f"Prices: {response.content}")]
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

def advisor(state: State):
    print("\n✅ Generating final recommendation...")
    
    response = llm.invoke(f"""
    You are a friendly PC buying advisor in Tunisia.
    Give a clear final recommendation based on the data below.
    
    User needs: PC for {state['use_case']}, budget {state['budget']} TND, in {state['user_location']}
    
    Recommended specs from research:
    {state['specs_research']}
    
    ACTUAL prices found in Tunisian stores:
    {state['local_prices']}
    
    IMPORTANT RULES:
    - Only recommend laptops/products ACTUALLY found in Tunisian stores above
    - Include the EXACT store URL where they can buy it
    - Use REAL prices — ignore monthly installment prices (3 mois, 6 mois, 9 mois, 12 mois)
    - The real price is always the HIGHEST number shown, not the monthly payment
    - If multiple options found, pick best value within budget of {state['budget']} TND
    - Never say "not available" if a store URL was found
    
    Format your answer as:

    🖥️ RECOMMENDED LAPTOP:
    <laptop name and model>

    💰 PRICE: <full price NOT monthly> TND

    🛒 BUY HERE: <exact store URL>

    ⚡ WHY: <simple explanation>

    🔄 ALTERNATIVES:
    <other options with prices and links>

    ⚠️ WATCH OUT FOR: <warnings>
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