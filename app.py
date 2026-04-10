# “””
app.py — Digital Behavior Coach (HF Space entry point)

FastAPI  ->  /reset  /step  /state   (OpenEnv validator — REQUIRED)
Gradio   ->  /ui                     (Interactive visual UI)
“””

import gradio as gr
import plotly.graph_objects as go
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from environment import DigitalCoachEnv, Action, Observation

# ──────────────────────────────────────────────────────────────────────────────

# FastAPI — OpenEnv validator endpoints (DO NOT REMOVE OR RENAME)

# ──────────────────────────────────────────────────────────────────────────────

api  = FastAPI(title=“Digital Behavior Coach — OpenEnv”)
_env = DigitalCoachEnv()

class ActionRequest(BaseModel):
action_type: Optional[str] = “do_nothing”
message:     Optional[str] = “”

@api.get(”/”)
def api_root():
return {“status”: “ok”, “environment”: “DigitalBehaviorCoach”}

@api.post(”/reset”)
def api_reset():
obs = _env.reset()
return JSONResponse(content=obs.model_dump())

@api.post(”/step”)
def api_step(action: ActionRequest):
obs, reward, done, info = _env.step(
Action(action_type=action.action_type, message=action.message)
)
return JSONResponse(content={
“observation”: obs.model_dump(),
“reward”:      reward.model_dump(),
“done”:        done,
“info”:        info,
})

@api.get(”/state”)
def api_state():
return JSONResponse(content=_env.state())

# ──────────────────────────────────────────────────────────────────────────────

# Gradio Visual UI

# ──────────────────────────────────────────────────────────────────────────────

_ui_env  = DigitalCoachEnv()
_history = []

ACTION_META = {
“send_reminder”: {“emoji”: “⏰”, “color”: “#f59e0b”, “cls”: “active-reminder”},
“block_app”:     {“emoji”: “🚫”, “color”: “#ef4444”, “cls”: “active-block”},
“encourage”:     {“emoji”: “💪”, “color”: “#10b981”, “cls”: “active-encourage”},
“do_nothing”:    {“emoji”: “😌”, “color”: “#6366f1”, “cls”: “active-nothing”},
}
COACH_LINES = {
“send_reminder”: “Hey — you’ve been on your phone a while. Time for a break?”,
“block_app”:     “That’s enough. I’m locking distractions for you right now.”,
“encourage”:     “You’re doing brilliant. Keep that streak alive! 🔥”,
“do_nothing”:    “You’re managing yourself well. I’ll stay out of your way.”,
}
GOAL_EM = {“focus”: “🎯”, “sleep”: “😴”, “balance”: “⚖️”}
APP_EM  = {“social”: “📱”, “entertainment”: “🎬”, “work”: “💼”}

def _bar(val: float, total: float = 1.0, w: int = 10) -> str:
f = max(0.0, min(1.0, val / total))
n = round(f * w)
return “█” * n + “░” * (w - n)

def _build_state_md(obs) -> str:
hr = obs.hour_of_day
if hr >= 22:   period = “🌙 Night”
elif hr >= 18: period = “🌆 Evening”
elif hr >= 8:  period = “☀️ Day”
else:          period = “🌃 Late Night”
gEm = GOAL_EM.get(obs.user_goal, “🎯”)
aEm = APP_EM.get(obs.app_category, “📲”)
return f”””## 📡 Environment State

|Field        |Value                            |Progress                           |
|-------------|---------------------------------|-----------------------------------|
|🕐 Time       |`{hr:02d}:00` {period}           |                                   |
|📱 Screen Time|`{obs.screen_time_today:.1f} hrs`|`{_bar(obs.screen_time_today, 12)}`|
|{gEm} Goal   |`{obs.user_goal.upper()}`        |                                   |
|🔥 Streak     |`{obs.streak_days} days`         |                                   |
|{aEm} App    |`{obs.app_category}`             |                                   |
|🧠 Habit Score|`{obs.habit_score:.2f}`          |`{_bar(obs.habit_score)}`          |
|🛡️ Resistance |`{obs.resistance_level:.2f}`     |`{_bar(obs.resistance_level)}`     |
|↩️ Last Effect|`{obs.last_action_effect}`       |                                   |

def _build_coach_md(action_type, reward_val, reason, done, step) -> str:
emoji = ACTION_META[action_type][“emoji”]
line  = COACH_LINES[action_type]
stars = “⭐” * max(1, round(reward_val * 5))
done_note = “\n\n—\n✅ **Episode complete!** Press **Reset** to start again.” if done else “”
return f”””## {emoji} Step {step} — `{action_type}`

> *”{line}”*

**Reward:** `{reward_val:.2f}` {stars}

**Reason:** {reason}{done_note}”””

def _build_history_md() -> str:
if not _history:
return “*No steps yet — press an action button below.*”
rows = [”| # | Action | Reward | Screen | Habit | Resist |”,
“|—|—|—|—|—|—|”]
for h in reversed(_history[-15:]):
em = ACTION_META[h[“action”]][“emoji”]
rows.append(
f”| {h[‘step’]} | {em} `{h['action']}` | `{h['reward']:.2f}` “
f”| `{h['screen_time']:.1f}h` | `{h['habit_score']:.2f}` | `{h['resistance']:.2f}` |”
)
return “\n”.join(rows)

def _build_chart():
dark = “#070d1a”
panel = “#0d1526”
grid  = “#1e3a5f”
if not _history:
fig = go.Figure()
fig.add_annotation(
text=“Take an action to see live performance stats”,
xref=“paper”, yref=“paper”, x=0.5, y=0.5,
showarrow=False, font=dict(size=14, color=”#334155”),
)
fig.update_layout(
paper_bgcolor=dark, plot_bgcolor=dark,
xaxis=dict(visible=False), yaxis=dict(visible=False),
margin=dict(l=20, r=20, t=20, b=20), height=300,
)
return fig

```
steps   = [h["step"]        for h in _history]
rewards = [h["reward"]      for h in _history]
screens = [h["screen_time"] for h in _history]
habits  = [h["habit_score"] for h in _history]
resists = [h["resistance"]  for h in _history]

fig = go.Figure()
fig.add_trace(go.Scatter(x=steps, y=rewards, name="Reward",
    line=dict(color="#10b981", width=2.5),
    fill="tozeroy", fillcolor="rgba(16,185,129,0.07)",
    mode="lines+markers", marker=dict(size=5)))
fig.add_trace(go.Scatter(x=steps, y=[s/12 for s in screens], name="Screen Time (÷12)",
    line=dict(color="#ef4444", width=2),
    mode="lines+markers", marker=dict(size=5)))
fig.add_trace(go.Scatter(x=steps, y=habits, name="Habit Score",
    line=dict(color="#38bdf8", width=2),
    mode="lines+markers", marker=dict(size=5)))
fig.add_trace(go.Scatter(x=steps, y=resists, name="Resistance",
    line=dict(color="#f59e0b", width=1.5, dash="dot"), mode="lines"))
fig.update_layout(
    paper_bgcolor=dark, plot_bgcolor=panel,
    font=dict(color="#94a3b8", family="monospace", size=11),
    legend=dict(bgcolor=panel, bordercolor=grid, borderwidth=1,
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(title="Step", gridcolor=grid, zerolinecolor=grid),
    yaxis=dict(title="Value", gridcolor=grid, zerolinecolor=grid, range=[0, 1.15]),
    margin=dict(l=50, r=20, t=40, b=40), height=320,
    hovermode="x unified",
)
return fig
```

def reset_ui():
global _history
_history = []
obs = _ui_env.reset()
return (
_build_state_md(obs), _build_chart(),
“## 🟢 Ready\n\nEnvironment reset. Choose an action to begin coaching.”,
_build_history_md(),
)

def take_action(action_type: str):
global _history
obs, reward, done, info = _ui_env.step(Action(action_type=action_type))
_history.append({
“step”: info[“step”], “action”: action_type,
“reward”: reward.value, “reason”: reward.reason,
“screen_time”: obs.screen_time_today,
“habit_score”: obs.habit_score, “resistance”: obs.resistance_level,
})
return (
_build_state_md(obs), _build_chart(),
_build_coach_md(action_type, reward.value, reward.reason, done, info[“step”]),
_build_history_md(),
)

CSS = “””
@import url(‘https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap’);

body, .gradio-container {
background: #020817 !important;
font-family: ‘JetBrains Mono’, monospace !important;
}
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }

.gr-markdown h1 {
font-family: ‘Syne’, sans-serif !important;
font-size: 1.9rem !important; font-weight: 800 !important;
background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #34d399 100%) !important;
-webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
}
.gr-markdown h2 {
font-family: ‘Syne’, sans-serif !important;
font-size: 0.95rem !important; color: #38bdf8 !important;
text-transform: uppercase !important; letter-spacing: 0.08em !important;
}
.gr-markdown h3 {
color: #64748b !important; font-size: 0.78rem !important;
text-transform: uppercase !important; letter-spacing: 0.12em !important;
}
.gr-markdown code {
background: #0d2847 !important; border: 1px solid #1e3a5f !important;
color: #34d399 !important; padding: 0.08rem 0.4rem !important;
border-radius: 5px !important; font-size: 0.85rem !important;
}
.gr-markdown blockquote {
border-left: 3px solid #818cf8 !important; background: #0d1526 !important;
padding: 0.55rem 1rem !important; border-radius: 0 8px 8px 0 !important;
color: #c4b5fd !important; font-style: italic !important; margin: 0.6rem 0 !important;
}
.gr-markdown table { border-collapse: collapse !important; width: 100% !important; font-size: 0.8rem !important; }
.gr-markdown th {
background: #0d1f38 !important; color: #38bdf8 !important;
padding: 0.4rem 0.65rem !important; border: 1px solid #1e3a5f !important;
text-transform: uppercase !important; font-size: 0.7rem !important; letter-spacing: 0.06em !important;
}
.gr-markdown td { padding: 0.38rem 0.65rem !important; border: 1px solid #0f2744 !important; color: #94a3b8 !important; }
.gr-markdown tr:hover td { background: #0d1f38 !important; }

button.lg {
background: #0d1526 !important; border: 1px solid #1e3a5f !important;
color: #94a3b8 !important; border-radius: 10px !important;
font-family: ‘JetBrains Mono’, monospace !important; font-weight: 600 !important;
transition: all 0.2s ease !important;
}
button.lg:hover {
transform: translateY(-2px) !important;
box-shadow: 0 0 20px rgba(56,189,248,0.2) !important;
border-color: #38bdf8 !important; color: #e2e8f0 !important;
}
button.primary {
background: linear-gradient(135deg, #1e3a5f, #0d2847) !important;
border: 1px solid #38bdf8 !important; color: #38bdf8 !important;
}
button.primary:hover {
background: linear-gradient(135deg, #1e4a7f, #1e3a5f) !important;
box-shadow: 0 0 28px rgba(56,189,248,0.3) !important;
}
“””

with gr.Blocks(
theme=gr.themes.Base(primary_hue=“blue”, neutral_hue=“slate”),
css=CSS,
title=“Digital Behavior Coach”,
) as demo:

```
gr.Markdown("""
```

# 🧠 Digital Behavior Coach

**OpenEnv RL Environment** — An AI that coaches users toward healthier screen habits.
“””)

```
with gr.Row():
    reset_btn = gr.Button("⟳  Reset Environment", variant="primary", size="lg", scale=1)
    gr.Markdown(
        "<div style='padding:0.6rem 0;color:#334155;font-size:0.78rem;'>"
        "Reset starts a new episode with a random user profile. Take actions to coach them.</div>",
        scale=4,
    )

with gr.Row():
    state_md = gr.Markdown("*Press Reset to initialise.*", label="")
    coach_md = gr.Markdown("*Awaiting first action...*",   label="")

chart = gr.Plot(show_label=False)

gr.Markdown("### 🎮 Choose an Agent Action")
with gr.Row():
    btn_r = gr.Button("⏰  Send Reminder", size="lg")
    btn_b = gr.Button("🚫  Block App",     size="lg")
    btn_e = gr.Button("💪  Encourage",     size="lg")
    btn_n = gr.Button("😌  Do Nothing",    size="lg")

gr.Markdown("### 📜 Episode Step History")
history_md = gr.Markdown("*No steps yet.*")

gr.Markdown(
    "<div style='text-align:center;color:#1e3a5f;font-size:0.72rem;"
    "margin-top:1rem;border-top:1px solid #0f2744;padding-top:0.6rem;'>"
    "screen-coach-rl · OpenEnv v1.0 · Tasks: easy / medium / hard</div>"
)

_out = [state_md, chart, coach_md, history_md]
reset_btn.click(reset_ui,                               outputs=_out)
btn_r.click(lambda: take_action("send_reminder"),       outputs=_out)
btn_b.click(lambda: take_action("block_app"),           outputs=_out)
btn_e.click(lambda: take_action("encourage"),           outputs=_out)
btn_n.click(lambda: take_action("do_nothing"),          outputs=_out)
```

# ──────────────────────────────────────────────────────────────────────────────

# Mount Gradio into FastAPI and launch

# ──────────────────────────────────────────────────────────────────────────────

app = gr.mount_gradio_app(api, demo, path=”/ui”)

if **name** == “**main**”:
uvicorn.run(app, host=“0.0.0.0”, port=7860)
