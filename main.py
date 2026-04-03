from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Digital Behavior Coach - OpenEnv")

class Action(BaseModel):
    action_type: str
    message: Optional[str] = ""

@app.get("/")
def root():
    return {"status": "ok", "environment": "DigitalBehaviorCoach"}

@app.post("/reset")
def reset():
    from environment import env
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: Action):
    from environment import env
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    from environment import env
    return env.state()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)