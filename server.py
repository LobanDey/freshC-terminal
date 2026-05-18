import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from google import genai
from pydantic import BaseModel

# Configuration
api_key = os.environ.get("GEMMA_API_KEY")
model_id = "gemma-7b" # Or "models/gemma-7b-it" depending on the version

if api_key:
    client = genai.Client(api_key=api_key)
else:
    client = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    code: str = ""

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not client:
        # Fallback to simulated response if no API key is provided
        return {"response": f"GEMMA_MOCK_RESPONSE: (No API key found) I see you have {len(req.code)} chars of code. You asked: {req.query}"}

    try:
        prompt = f"Context code:\n```\n{req.code}\n```\n\nUser query: {req.query}\n\nAssistant:"
        response = client.models.generate_content(
            model=model_id,
            contents=prompt
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
