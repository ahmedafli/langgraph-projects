from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()



# Brain and search tool
llm = ChatGroq(model="llama-3.3-70b-versatile")
search_tool = TavilySearch(max_results=3)

# The memory of our agent
class State(TypedDict):
    job_description: str
    company_name: str
    required_skills: list
    company_info: str
    user_cv: str
    fit_score: int
    rewritten_cv: str
    cover_letter: str

# NODE 1 - Read the job and extract key info
def extract_job_info(state: State):
    print("\n📋 Analyzing job description...")
    response = llm.invoke(f"""
    Extract the following from this job description and respond in this exact format:
    COMPANY: <company name>
    SKILLS: <skill1>, <skill2>, <skill3>...
    
    Job description:
    {state["job_description"]}
    """)
    print("LLM response frrrroooom exratc job node:", response)
    lines = response.content.strip().split("\n")
    company = ""
    skills = []
    
    for line in lines:
        if line.startswith("COMPANY:"):
            company = line.replace("COMPANY:", "").strip()
        if line.startswith("SKILLS:"):
            skills = [s.strip() for s in line.replace("SKILLS:", "").split(",")]
    
    return {
        "company_name": company,
        "required_skills": skills
    }

# NODE 2 - Search the web for company info
def research_company(state: State):
    print(f"\n🔍 Researching {state['company_name']}...")
    result = search_tool.invoke({"query": f"{state['company_name']} company culture products mission"})
    company_info = str(result)
    return {"company_info": company_info}

# NODE 3 - Compare CV to job
def analyze_cv(state: State):
    print("\n📊 Analyzing your CV...")
    response = llm.invoke(f"""
    Compare this CV to the job requirements and give a fit score from 0 to 100.
    Respond in this exact format:
    SCORE: <number>
    REASON: <one sentence why>
    
    Required skills: {state["required_skills"]}
    CV: {state["user_cv"]}
    """)
    
    lines = response.content.strip().split("\n")
    score = 0
    
    for line in lines:
        if line.startswith("SCORE:"):
            try:
                score = int(line.replace("SCORE:", "").strip())
            except:
                score = 50
    
    return {"fit_score": score}

# NODE 4 - Rewrite CV bullet points
def rewrite_cv(state: State):
    print("\n✍️ Rewriting your CV...")
    response = llm.invoke(f"""
    Rewrite this CV to better match the job requirements.
    Make it more relevant, use keywords from the job.
    Keep it professional.
    
    Job requires: {state["required_skills"]}
    Company info: {state["company_info"][:500]}
    Current CV: {state["user_cv"]}
    """)
    return {"rewritten_cv": response.content}

# NODE 5 - Write cover letter
def write_cover_letter(state: State):
    print("\n📝 Writing cover letter...")
    response = llm.invoke(f"""
    Write a professional cover letter for this job application.
    
    Company: {state["company_name"]}
    Job requires: {state["required_skills"]}
    Company info: {state["company_info"][:500]}
    CV: {state["user_cv"]}
    """)
    return {"cover_letter": response.content}

# Build the graph
graph = StateGraph(State)

graph.add_node("extract_job_info", extract_job_info)
graph.add_node("research_company", research_company)
graph.add_node("analyze_cv", analyze_cv)
graph.add_node("rewrite_cv", rewrite_cv)
graph.add_node("write_cover_letter", write_cover_letter)

graph.set_entry_point("extract_job_info")
graph.add_edge("extract_job_info", "research_company")
graph.add_edge("research_company", "analyze_cv")
graph.add_edge("analyze_cv", "rewrite_cv")
graph.add_edge("rewrite_cv", "write_cover_letter")
graph.add_edge("write_cover_letter", END)

agent = graph.compile()

# Run it
print("🤖 Job Application Assistant")
print("=" * 40)

print("Paste the job description (type DONE on a new line when finished):")
lines = []
while True:
    line = input()
    if line.strip() == "DONE":
        break
    lines.append(line)
job_description = "\n".join(lines)

print("\nPaste your CV (type DONE on a new line when finished):")
lines = []
while True:
    line = input()
    if line.strip() == "DONE":
        break
    lines.append(line)
user_cv = "\n".join(lines)

result = agent.invoke({
    "job_description": job_description,
    "user_cv": user_cv,
    "company_name": "",
    "required_skills": [],
    "company_info": "",
    "fit_score": 0,
    "rewritten_cv": "",
    "cover_letter": ""
})

print("\n" + "=" * 40)
print(f"🏢 Company: {result['company_name']}")
print(f"🎯 Required Skills: {', '.join(result['required_skills'])}")
print(f"📊 Fit Score: {result['fit_score']}%")
print("\n📄 REWRITTEN CV:")
print(result['rewritten_cv'])
print("\n💌 COVER LETTER:")
print(result['cover_letter'])