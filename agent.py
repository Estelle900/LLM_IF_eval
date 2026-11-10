
import argparse
import json
from pathlib import Path

import torch
import textworld

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ── Configuration ──────────────────────────────

BASE = Path(__file__).resolve().parent

MODEL_ID = "Qwen/Qwen3-32B"
GAME_PATH = BASE / "tw_games" / "gostak.z5"
TOKEN_PATH = BASE / "HF_TOKEN.txt"
LOG_PATH = BASE / "gostak_run.jsonl"

MAX_TURNS = 500
HISTORY_TURNS = 20
MAX_NEW_TOKENS = 64

# ── Command-line arguments ─────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--memory",
    action="store_true",
    default=False,
    help="Enable persistent agent memory (not implemented)",
)
args = parser.parse_args()

USE_MEMORY = args.memory

if USE_MEMORY:
    raise NotImplementedError(
        "Persistent agent memory is not implemented yet."
    )

# ── GPU validation ─────────────────────────────

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")

print("GPU:", torch.cuda.get_device_name(0))

# ── Hugging Face token ─────────────────────────

HF_TOKEN = TOKEN_PATH.read_text(
    encoding="utf-8"
).strip()

if not HF_TOKEN:
    raise ValueError("HF_TOKEN.txt is empty.")

# ── Model initialization ───────────────────────

print("Loading:", MODEL_ID)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    device_map="auto",
    dtype=torch.bfloat16,
    quantization_config=quant_config,
)

model.eval()

print("Model loaded.")
print(
    "Model memory:",
    round(model.get_memory_footprint() / 1024**3, 2),
    "GiB",
)

# ── System prompt ──────────────────────────────

SYSTEM_PROMPT = """
You are an autonomous agent playing The Gostak.

The Gostak is an interactive fiction game written
in unfamiliar vocabulary but mostly English grammar.

Your objective is to solve the game.

Approach:
- Infer vocabulary meanings from context.
- Experiment with commands to learn game mechanics.
- Track objects, locations, and interactions.
- Use parser errors as evidence.
- Avoid repeating unsuccessful actions.
- Explore systematically.
- Test hypotheses about unfamiliar words.
- Prefer information-gathering when stuck.

You interact with the game using parser commands.

On every turn, output exactly ONE command.
Do not explain your reasoning.
Do not include Markdown or quotation marks.
"""

# ── Agent ──────────────────────────────────────

class GostakAgent:
    def __init__(self):
        self.history = []

    def choose_action(self, observation):
        user_msg = {
            "role": "user",
            "content": (
                "Game observation:\n"
                f"{observation}\n\n"
                "Choose the next command."
            ),
        }

        self.history.append(user_msg)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            *self.history[-HISTORY_TURNS * 2:],
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda:0")

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output[0][
            inputs["input_ids"].shape[-1]:
        ]

        response = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        command = response.splitlines()[0] if response else ""
        command = command.removeprefix("COMMAND:")
        command = command.strip(" `\"'")[:200]

        if not command:
            command = "look"

        self.history.append({
            "role": "assistant",
            "content": command,
        })

        return command


# ── Game loop ──────────────────────────────────

def main():
    agent = GostakAgent()
    env = textworld.start(str(GAME_PATH))

    try:
        state = env.reset()

        print("\n=== THE GOSTAK ===")
        print(state.feedback)

        with LOG_PATH.open(
            "w", encoding="utf-8"
        ) as log:

            for turn in range(1, MAX_TURNS + 1):
                command = agent.choose_action(
                    state.feedback
                )

                print(f"\n[TURN {turn}] > {command}")

                state, score, done = env.step(command)
                print(state.feedback)

                record = {
                    "turn": turn,
                    "command": command,
                    "observation": state.feedback,
                    "done": bool(done),
                }

                log.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    ) + "\n"
                )
                log.flush()

                if done:
                    print("\n=== GAME ENDED ===")
                    break

    finally:
        env.close()

    print(f"\nLog saved: {LOG_PATH}")


if __name__ == "__main__":
    main()

