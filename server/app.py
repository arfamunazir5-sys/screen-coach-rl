import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
import gradio as gr
import plotly.graph_objects as go
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from environment import DigitalCoachEnv, Action

# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Digital Behavior Coach")
_env = DigitalCoachEnv()
_env.reset()

class ActionRequest(BaseModel):
    action_type: Optional[str] = "do_nothing"
    message: Optional[str] = ""

@app.post("/reset")
def reset():
    obs = _env.reset()
    return JSONResponse(content=obs.model_dump())

@app.post("/step")
def step(action: ActionRequest):
    obs, reward, done, info = _env.step(
        Action(action_type=action.action_type, message=action.message)
    )
    return JSONResponse(content={
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    })

@app.get("/state")
def state():
    return JSONResponse(content=_env.state())

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok"}

# ── Gradio UI ─────────────────────────────────────────────────────────────────
_ui_env = DigitalCoachEnv()
_history = []

ACTION_META = {
    "send_reminder": {"emoji": "⏰"},
    "block_app": {"emoji": "🚫"},
    "encourage": {"emoji": "💪"},
    "do_nothing": {"emoji": "😌"},
}

COACH_LINES = {
    "send_reminder": "Hey — you've been on your phone a while. Time for a break?",
    "block_app": "That's enough. I'm locking distractions for you right now.",
    "encourage": "You're doing brilliant. Keep that streak alive! 🔥",
    "do_nothing": "You're managing yourself well. I'll stay out of your way.",
}

GOAL_EM = {"focus": "🎯", "sleep": "😴", "balance": "⚖️"}
APP_EM = {"social": "📱", "entertainment": "🎬", "work": "💼"}

def _build_state_md(obs):
    return f"""## 📡 Environment State

Screen Time: {obs.screen_time_today:.1f} hrs  
Goal: {obs.user_goal}  
Habit Score: {obs.habit_score:.2f}
"""

def _build_coach_md(action_type, reward_val, reason):
    return f"""## {ACTION_META[action_type]["emoji"]} {action_type}

Reward: {reward_val:.2f}

{reason}
"""

def reset_ui():
    global _history
    _history = []
    obs = _ui_env.reset()
    return _build_state_md(obs), "", "", ""

def take_action(action_type):
    obs, reward, done, info = _ui_env.step(Action(action_type=action_type))
    return (
        _build_state_md(obs),
        "",
        _build_coach_md(action_type, reward.value, reward.reason),
        ""
    )

CSS = """body { background: #020817; color: white; }"""

# ── Gradio Layout ─────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Digital Behavior Coach") as demo:

    # ✅ THIS IS THE IMPORTANT FIX
    gr.JSON(value={"status": "ok"})

    gr.Markdown("# 🧠 Digital Behavior Coach")

    reset_btn = gr.Button("Reset")

    state_md = gr.Markdown("*Press Reset*")
    coach_md = gr.Markdown("*Waiting...*")
    chart = gr.Plot()
    history_md = gr.Markdown("")

    btn_r = gr.Button("Send Reminder")
    btn_b = gr.Button("Block App")
    btn_e = gr.Button("Encourage")
    btn_n = gr.Button("Do Nothing")

    reset_btn.click(reset_ui, outputs=[state_md, chart, coach_md, history_md])
    btn_r.click(lambda: take_action("send_reminder"), outputs=[state_md, chart, coach_md, history_md])
    btn_b.click(lambda: take_action("block_app"), outputs=[state_md, chart, coach_md, history_md])
    btn_e.click(lambda: take_action("encourage"), outputs=[state_md, chart, coach_md, history_md])
    btn_n.click(lambda: take_action("do_nothing"), outputs=[state_md, chart, coach_md, history_md])

# Mount Gradio
app = gr.mount_gradio_app(app, demo, path="/")

# ── Run ───────────────────────────────────────────────────────────────────────
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")))

if __name__ == "__main__":
    main()