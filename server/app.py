import os
import sys

# Add parent directory to path so we can import environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from environment import DigitalCoachEnv, Action
import uvicorn

app = FastAPI(title="Digital Behavior Coach")
_env = DigitalCoachEnv()
_env.reset()


class ActionRequest(BaseModel):
    action_type: Optional[str] = "do_nothing"
    message: Optional[str] = ""


@app.get("/")
def root():
    return {"status": "ok", "environment": "DigitalBehaviorCoach"}


@app.post("/reset")
def reset():
    obs = _env.reset()
    return JSONResponse(content=obs.model_dump())


@app.post("/step")
def step(action: ActionRequest):
    try:
        obs, reward, done, info = _env.step(
            Action(action_type=action.action_type, message=action.message)
        )
        return JSONResponse(content={
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        })
    except Exception:
        _env.reset()
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


def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860"))
    )


if __name__ == "__main__":
    main()