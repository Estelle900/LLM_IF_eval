# Testing LLMs' ability to play parser IF

## Quick test:

Start the server:

```bash
uvicorn server:app --host 127.0.0.1 --port 8000

curl -X POST http://127.0.0.1:8000/reset

curl -X POST http://127.0.0.1:8000/step \
  -H "Content-Type: application/json" \
  -d '{"command": "look"}'

```

Run the agent:

```bash
python agent.py
```
