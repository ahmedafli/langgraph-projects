import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from typing import Optional, TypedDict
from dotenv import load_dotenv
from database import SessionLocal, LearningGoal, DailyTask
from datetime import datetime, timedelta

import json

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")
search = TavilySearch(max_results=3)

# The shared state across all nodes
class PlannerState(TypedDict):
    goal: str
    topics: str
    hours_per_day: int
    weeks: int
    search_results: str
    curriculum: list
    goal_id: Optional[int]  # None at start, filled after save

# NODE 0 - Extract clean topics from messy user input
def extract_topics(state: PlannerState):
    print("\n🧹 Extracting clean topics from your input...")
    
    response = llm.invoke(f"""
    Extract a clean, comma-separated list of specific technologies/topics 
    from this user input. Be specific and use proper technical names.
    
    User's goal: {state['goal']}
    User's topics input: {state['topics']}
    
    Return ONLY a comma-separated list, nothing else.
    Example output format: TopicA, TopicB, TopicC, TopicD
    """)
    print(f"ddd Extract topics response AAAAAAAAAAAAA: {response}")
    clean_topics = response.content.strip()
    print(f"✅ddddddddddddddd Clean topics: {clean_topics}")
    
    return {"topics": clean_topics}    

# NODE 1 - Search for best learning resources
def search_resources(state: PlannerState):
    print("\n🔍 Searching for best learning path...")
    
    # Let LLM understand HOW topics relate to each other based on the goal
    strategy_response = llm.invoke(f"""
    Based on this goal and topics, determine the learning strategy.
    
    Goal: {state['goal']}
    Topics: {state['topics']}
    
    For each topic, decide:
    - Is it the MAIN focus or a SUPPORTING tool?
    - What's the best search query to find quality resources for it 
      GIVEN its role in the user's goal?
    
    Return as JSON array:
    [
        {{"topic": "TopicName", "role": "main/supporting", "search_query": "specific search query"}}
    ]
    Return ONLY the JSON, nothing else.
    """)
    
    content = strategy_response.content.strip()
    print(f"node 1111111 content = strategy_responsepointcontentpoint stripped: {content}")
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    strategy = json.loads(content)
    print(f"📋 Search strategy: {strategy}")
    
    all_results = []
    for item in strategy:
        results = search.invoke({"query": item["search_query"]})
        all_results.extend(results['results'])
    
    
    formatted = ""
    for r in all_results:
        formatted += f"\nSOURCE: {r['url']}\nTITLE: {r['title']}\nCONTENT: {r['content'][:300]}\n---"
    print(f"📚 Formatted search results: {formatted}")
    
    return {"search_results": formatted}

# NODE 2 - Generate the curriculum
def generate_curriculum(state: PlannerState):
    print("\n📚 Generating curriculum...")
    total_days = state['weeks'] * 7
    
    def generate(max_days):
        response = llm.invoke(f"""
    You are an expert learning coach. Create a detailed day-by-day learning plan.
    
    Goal: {state['goal']}
    Topics to learn: {state['topics']}
    Hours per day: {state['hours_per_day']}
    Total days: {total_days}
    
    Based on these resources:
    {state['search_results']}
    
    Create a structured learning plan covering ONLY the topics provided above.
    Return ONLY a JSON array with this exact structure:
    [
        {{
            "day": <number>,
            "topic": "<one of the topics from the list above>",
            "subtopic": "<specific subtopic for that day>",
            "tasks": ["<task1>", "<task2>", "<task3>"],
            "estimated_hours": <number>,
            "resources": [
                {{"title": "<title from search results>", "url": "<url from search results>"}},
                {{"title": "<title from search results>", "url": "<url from search results>"}}
            ]
        }}
    ]
    
    RULES:
    - Cover ALL topics mentioned: {state['topics']}
    - Distribute days proportionally across all topics
    - Each day max {state['hours_per_day']} hours
    - Start simple, gradually increase complexity
    - Include practice days every 5 days
    - Last 3 days are for a final project combining all topics
    - For resources: ONLY use titles and URLs from the search results provided above
    - Each day should have 1-3 relevant resources matching that day's topic
    - Never invent or hallucinate URLs
    - Return ONLY the JSON array, no other text, no markdown
    """)
        return response.content.strip()
    
    # Try with full days first
    content = generate(total_days)
    
    # Clean markdown if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    # Try parsing
    try:
        curriculum = json.loads(content)
        print(f"✅ Generated {len(curriculum)} days")
        return {"curriculum": curriculum}
    
    except Exception as e:
        print(f"⚠️ First attempt failed: {e}")
        print(f"🔄 Retrying with fewer days...")
        
        # Retry with half the days
        content = generate(total_days // 2)
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        try:
            curriculum = json.loads(content)
            print(f"✅ Generated {len(curriculum)} days on retry")
            return {"curriculum": curriculum}
        
        except Exception as e2:
            print(f"❌ Second attempt failed: {e2}")
            return {"curriculum": []}


# NODE - Save approved curriculum to database
def save_to_db(state: PlannerState):
    print("\n💾 Saving to database...")
    
    db = SessionLocal()
    
    try:
        # Create the goal record
        new_goal = LearningGoal(
            goal=state['goal'],
            topics=state['topics'],
            hours_per_day=state['hours_per_day'],
            target_end_date=datetime.now() + timedelta(weeks=state['weeks'])
        )
        db.add(new_goal)
        db.commit()
        db.refresh(new_goal)
        
        print(f"✅ Goal saved with ID: {new_goal.id}")
        
        # Create each daily task
        for day in state['curriculum']:
            task = DailyTask(
                goal_id=new_goal.id,
                day_number=day['day'],
                topic=day['topic'],
                subtopic=day['subtopic'],
                instructions=", ".join(day.get('tasks', [])),
                resources=json.dumps(day.get('resources', [])),  # save as JSON string
                estimated_hours=day['estimated_hours'],
                scheduled_date=datetime.now() + timedelta(days=day['day']-1)
            )
            db.add(task)
        
        db.commit()
        print(f"✅ Saved {len(state['curriculum'])} daily tasks")
        return {
            "goal_id": new_goal.id,
            "curriculum": state["curriculum"]  # keep it in state
        }
        
    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        print(f"❌ Full error: {repr(e)}")  # ADD THIS
        db.rollback()
        return {"goal_id": None, "curriculum": state["curriculum"]}
    finally:
        db.close()
    

# Build the graph
graph = StateGraph(PlannerState)

graph.add_node("extract_topics", extract_topics)
graph.add_node("search_resources", search_resources)
graph.add_node("generate_curriculum", generate_curriculum)
graph.add_node("save_to_db", save_to_db)

graph.set_entry_point("extract_topics")
graph.add_edge("extract_topics", "search_resources")
graph.add_edge("search_resources", "generate_curriculum")
graph.add_edge("generate_curriculum", "save_to_db")
graph.add_edge("save_to_db", END)

planner_agent = graph.compile()

# Called by FastAPI
def run_planner(goal: str, topics: str, hours_per_day: int, weeks: int):
    result = planner_agent.invoke({
    "goal": goal,
    "topics": topics,
    "hours_per_day": hours_per_day,
    "weeks": weeks,
    "search_results": "",
    "curriculum": [],
    "goal_id": None  # empty until save_to_db fills it
    })
    return {
        "goal_id": result["goal_id"],
        "total_days": len(result["curriculum"])
    }

if __name__ == "__main__":
    print("🎯 Learning Plan Creator\n")
    
    goal = input("What's your goal?: ")
    topics = input("What topics? (comma separated): ")
    hours_per_day = int(input("Hours per day?: "))
    weeks = int(input("How many weeks?: "))
    
    result = planner_agent.invoke({
    "goal": goal,
    "topics": topics,
    "hours_per_day": hours_per_day,
    "weeks": weeks,
    "search_results": "",
    "curriculum": [],
    "goal_id": None  # empty until save_to_db fills it
    })
    
    