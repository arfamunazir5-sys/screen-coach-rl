"""
server/app.py — OpenEnv FastAPI server (required by openenv validate)
Exposes /reset, /step, /state endpoints for the OpenEnv validator.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import DigitalCoachEnv, Action

app = FastAPI(title="Digital Behavior Coach — OpenEnv")

env = DigitalCoachEnv()
env.reset()


class ActionRequest(BaseModel):
    action_type: Optional[str] = "do_nothing"
    message:     Optional[str] = ""


@app.get("/")
def root():
    return {"status": "ok", "environment": "DigitalBehaviorCoach"}


@app.post("/reset")
def reset():
    obs = env.reset()
    return JSONResponse(content=obs.model_dump())


@app.post("/step")
def step(action: ActionRequest):
    try:
        obs, reward, done, info = env.step(
            Action(action_type=action.action_type, message=action.message)
        )
        return JSONResponse(content={
            "observation": obs.model_dump(),
            "reward":      reward.model_dump(),
            "done":        done,
            "info":        info,
        })
    except Exception:
        env.reset()
        obs, reward, done, info = env.step(
            Action(action_type=action.action_type, message=action.message)
        )
        return JSONResponse(content={
            "observation": obs.model_dump(),
            "reward":      reward.model_dump(),
            "done":        done,
            "info":        info,
        })


@app.get("/state")
def state():
    return JSONResponse(content=env.state())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
