from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from threading import Lock
import textworld

app = FastAPI(title="The Gostak Environment")

env = textworld.start("tw_games/gostak.z5")
lock = Lock()
started = False
turn = 0


class Action(BaseModel):
    command: str = Field(min_length=1, max_length=200)


@app.post("/reset")
def reset():
    global started, turn

    with lock:
        state = env.reset()
        started = True
        turn = 0

        return {
            "observation": state.feedback,
            "done": False,
            "turn": turn,
        }


@app.post("/step")
def step(action: Action):
    global turn

    with lock:
        if not started:
            raise HTTPException(400, "Call /reset first")

        state, score, done = env.step(action.command)
        turn += 1

        return {
            "observation": state.feedback,
            "done": bool(done),
            "turn": turn,
        }

