from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
from environment import DigitalCoachEnv, Action

app = FastAPI(title="Digital Behavior Coach - OpenEnv")

env = DigitalCoachEnv()


class ActionRequest(BaseModel):
    action_type: str
    message: Optional[str] = ""


@app.get("/")
def root():
    return {"status": "ok", "environment": "DigitalBehaviorCoach"}


@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: ActionRequest):
    obs, reward, done, info = env.step(Action(action_type=action.action_type, message=action.message))
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    return env.state()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
