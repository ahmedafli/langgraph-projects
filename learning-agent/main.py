from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import SessionLocal, LearningGoal, DailyTask, Progress
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

load_dotenv()

app = FastAPI(title="Learning Agent API", version="1.0.0")

# Allow frontend to talk to API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGroq(model="llama-3.3-70b-versatile")
search = TavilySearch(max_results=3)

# ============ SCHEMAS ============

class CreateGoalRequest(BaseModel):
    goal: str
    topics: str
    hours_per_day: int
    weeks: int

class SubmitQuizRequest(BaseModel):
    task_id: int
    answers: list[str]  # ["A", "B", "C", "D"]
    questions: list[dict]  # the questions with correct answers

# ============ ROUTES ============

@app.get("/")
def root():
    return {"message": "Learning Agent API is running! 🚀"}

# --- GOALS ---

@app.post("/goals/create")
def create_goal(request: CreateGoalRequest):
    """Run the planner agent and save curriculum to database"""
    try:
        # Import planner logic
        from agents.planner import run_planner
        result = run_planner(
            goal=request.goal,
            topics=request.topics,
            hours_per_day=request.hours_per_day,
            weeks=request.weeks
        )
        return {"success": True, "goal_id": result["goal_id"], "total_days": result["total_days"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/goals")
def get_all_goals():
    """Get all learning goals"""
    db = SessionLocal()
    try:
        goals = db.query(LearningGoal).all()
        return [{
            "id": g.id,
            "goal": g.goal,
            "topics": g.topics,
            "hours_per_day": g.hours_per_day,
            "status": g.status,
            "start_date": g.start_date,
            "target_end_date": g.target_end_date
        } for g in goals]
    finally:
        db.close()

@app.get("/goals/{goal_id}")
def get_goal(goal_id: int):
    """Get one goal with all its tasks"""
    db = SessionLocal()
    try:
        goal = db.query(LearningGoal).filter(LearningGoal.id == goal_id).first()
        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")
        
        tasks = db.query(DailyTask).filter(DailyTask.goal_id == goal_id).all()
        completed = len([t for t in tasks if t.status == "completed"])
        
        return {
            "id": goal.id,
            "goal": goal.goal,
            "topics": goal.topics,
            "hours_per_day": goal.hours_per_day,
            "status": goal.status,
            "start_date": goal.start_date,
            "target_end_date": goal.target_end_date,
            "total_tasks": len(tasks),
            "completed_tasks": completed,
            "progress_percentage": round((completed / len(tasks)) * 100) if tasks else 0
        }
    finally:
        db.close()

# --- DAILY ---

@app.get("/daily/{goal_id}")
def get_todays_task(goal_id: int):
    """Fetch today's pending task"""
    db = SessionLocal()
    try:
        task = db.query(DailyTask).filter(
            DailyTask.goal_id == goal_id,
            DailyTask.status == "pending"
        ).order_by(DailyTask.day_number).first()
        
        if not task:
            return {"message": "🎉 You completed the entire curriculum!"}
        
        return {
            "id": task.id,
            "day": task.day_number,
            "topic": task.topic,
            "subtopic": task.subtopic,
            "resources": task.resources,
            "resources_urls": json.loads(task.resources_urls) if task.resources_urls else [],
            "estimated_hours": task.estimated_hours,
            "status": task.status
        }
    finally:
        db.close()

@app.post("/daily/quiz/{task_id}")
def generate_quiz(task_id: int):
    """Generate quiz questions for a task"""
    db = SessionLocal()
    try:
        task = db.query(DailyTask).filter(DailyTask.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        response = llm.invoke(f"""
        Generate 4 multiple choice questions to test understanding of:
        
        Topic: {task.topic}
        Subtopic: {task.subtopic}
        What was learned: {task.resources}
        
        Return ONLY a JSON array:
        [
            {{
                "question": "<question text>",
                "options": {{
                    "A": "<option A>",
                    "B": "<option B>",
                    "C": "<option C>",
                    "D": "<option D>"
                }},
                "correct": "<A, B, C, or D>",
                "explanation": "<why this is correct>"
            }}
        ]
        Return ONLY the JSON array, no markdown.
        """)
        
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        questions = json.loads(content)
        return {"task_id": task_id, "questions": questions}
    
    finally:
        db.close()

@app.post("/daily/submit")
def submit_quiz(request: SubmitQuizRequest):
    """Score quiz and save progress"""
    db = SessionLocal()
    try:
        task = db.query(DailyTask).filter(DailyTask.id == request.task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Score the quiz
        correct = 0
        feedback = []
        
        for i, (q, answer) in enumerate(zip(request.questions, request.answers)):
            is_correct = answer.upper() == q['correct'].upper()
            if is_correct:
                correct += 1
            feedback.append({
                "question": i + 1,
                "correct": is_correct,
                "your_answer": answer,
                "correct_answer": q['correct'],
                "explanation": q.get('explanation', '')
            })
        
        score = (correct / len(request.questions)) * 100
        passed = score >= 70
        
        # Update task status
        task.status = "completed" if passed else "needs_review"
        
        # Save progress
        progress = Progress(
            task_id=request.task_id,
            completed_at=datetime.now(),
            quiz_score=score,
            notes=str(feedback),
            adjusted=not passed
        )
        db.add(progress)
        db.commit()
        
        return {
            "score": score,
            "passed": passed,
            "correct": correct,
            "total": len(request.questions),
            "feedback": feedback,
            "message": "✅ Task completed! Tomorrow's task is ready." if passed else "❌ Review needed. You'll redo this topic."
        }
    finally:
        db.close()

# --- PROGRESS ---

@app.get("/progress/{goal_id}")
def get_progress(goal_id: int):
    """Get full progress for a goal"""
    db = SessionLocal()
    try:
        tasks = db.query(DailyTask).filter(DailyTask.goal_id == goal_id).all()
        completed = [t for t in tasks if t.status == "completed"]
        pending = [t for t in tasks if t.status == "pending"]
        review = [t for t in tasks if t.status == "needs_review"]
        
        progress_records = db.query(Progress).join(DailyTask).filter(
            DailyTask.goal_id == goal_id
        ).all()
        
        avg_score = sum(p.quiz_score for p in progress_records) / len(progress_records) if progress_records else 0
        
        return {
            "total_tasks": len(tasks),
            "completed": len(completed),
            "pending": len(pending),
            "needs_review": len(review),
            "progress_percentage": round((len(completed) / len(tasks)) * 100) if tasks else 0,
            "average_quiz_score": round(avg_score, 1),
            "completed_topics": [t.topic for t in completed]
        }
    finally:
        db.close()