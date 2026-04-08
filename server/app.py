import gradio as gr
import json
import plotly.graph_objects as go
from environment import DigitalCoachEnv, Action

env = DigitalCoachEnv()
history = []

def reset_env():
    global history
    history = []
    obs = env.reset()
    return (
        f"Goal: {obs.user_goal} | Hour: {obs.hour_of_day} | Screen Time: {obs.screen_time_today}hrs | Resistance: {obs.resistance_level} | Habit Score: {obs.habit_score}",
        create_chart(),
        ""
    )

def take_action(action_type):
    global history
    action = Action(action_type=action_type)
    obs, reward, done, info = env.step(action)
    history.append({
        "step": info["step"],
        "action": action_type,
        "reward": reward.value,
        "screen_time": obs.screen_time_today,
        "habit_score": obs.habit_score,
        "resistance": obs.resistance_level
    })
    status = f"Goal: {obs.user_goal} | Hour: {obs.hour_of_day} | Screen Time: {obs.screen_time_today}hrs | Resistance: {obs.resistance_level} | Habit Score: {obs.habit_score}"
    log = f"Step {info['step']}: {action_type} → Reward: {reward.value} | {reward.reason}"
    return status, create_chart(), log

def create_chart():
    if not history:
        fig = go.Figure()
        fig.update_layout(title="Run the environment to see stats", template="plotly_dark")
        return fig

    steps = [h["step"] for h in history]
    rewards = [h["reward"] for h in history]
    screen_times = [h["screen_time"] for h in history]
    habit_scores = [h["habit_score"] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=rewards, name="Reward", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=steps, y=screen_times, name="Screen Time (hrs)", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=steps, y=habit_scores, name="Habit Score", line=dict(color="blue")))
    fig.update_layout(
        title="Digital Behavior Coach — Live Stats",
        xaxis_title="Step",
        yaxis_title="Value",
        template="plotly_dark"
    )
    return fig

with gr.Blocks(theme=gr.themes.Soft(), title="Digital Behavior Coach") as demo:
    gr.Markdown("# 🧠 Digital Behavior Coach")
    gr.Markdown("An RL environment that coaches users toward healthier screen habits.")

    with gr.Row():
        reset_btn = gr.Button("🔄 Reset Environment", variant="primary")

    status_box = gr.Textbox(label="Current State", interactive=False)
    log_box = gr.Textbox(label="Last Action Result", interactive=False)
    chart = gr.Plot(label="Live Dashboard")

    with gr.Row():
        gr.Button("📱 Send Reminder").click(lambda: take_action("send_reminder"), outputs=[status_box, chart, log_box])
        gr.Button("🚫 Block App").click(lambda: take_action("block_app"), outputs=[status_box, chart, log_box])
        gr.Button("💪 Encourage").click(lambda: take_action("encourage"), outputs=[status_box, chart, log_box])
        gr.Button("⏸ Do Nothing").click(lambda: take_action("do_nothing"), outputs=[status_box, chart, log_box])

    reset_btn.click(reset_env, outputs=[status_box, chart, log_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
