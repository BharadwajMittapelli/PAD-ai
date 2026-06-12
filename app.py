"""PAD.ai - Simple App"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import os

print("=" * 50)
print("PAD.ai STARTING - THIS SHOULD BE VISIBLE!")
print("=" * 50)

# Static directory is in same folder as this file
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
print(f"STATIC_DIR = {STATIC_DIR}")
print(f"EXISTS = {os.path.exists(STATIC_DIR)}")

app = FastAPI(title="PAD.ai")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    print("Static files mounted!")

class Request(BaseModel):
    url: str
    email_content: Optional[str] = None

class Response(BaseModel):
    is_phishing: bool
    confidence: float
    features_analyzed: dict

@app.get("/", response_class=HTMLResponse)
async def home():
    print("HOME ROUTE CALLED!")
    path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>PAD.ai - Static files not found</h1>")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=Response)
def predict(req: Request):
    is_phish = "phishing" in req.url.lower()
    return Response(
        is_phishing=is_phish,
        confidence=0.95 if is_phish else 0.05,
        features_analyzed={"url_length": len(req.url)}
    )
