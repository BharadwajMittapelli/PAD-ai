"""
PAD.ai - Phishing Attack Detection API
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import os

print("[PAD.ai] ====== STARTING SERVER ======")

# Get absolute path to static directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

print(f"[PAD.ai] Script dir: {SCRIPT_DIR}")
print(f"[PAD.ai] Project root: {PROJECT_ROOT}")
print(f"[PAD.ai] Static dir: {STATIC_DIR}")
print(f"[PAD.ai] Static exists: {os.path.exists(STATIC_DIR)}")

if os.path.exists(STATIC_DIR):
    files = os.listdir(STATIC_DIR)
    print(f"[PAD.ai] Static files: {files}")

app = FastAPI(title="PAD.ai - Phishing Attack Detection")

# Mount static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    print("[PAD.ai] Static files mounted at /static")

class PredictionRequest(BaseModel):
    url: str
    email_content: Optional[str] = None
    sender: Optional[str] = None

class PredictionResponse(BaseModel):
    is_phishing: bool
    confidence: float
    features_analyzed: dict

def detect_phishing(url: str, email_content: str = None):
    """Simple heuristic for phishing detection."""
    url_lower = url.lower()
    
    suspicious = ["phishing", "suspicious", "malware", "login-verify"]
    if any(kw in url_lower for kw in suspicious):
        return True, 0.95
    
    safe = ["google.com", "microsoft.com", "github.com"]
    if any(d in url_lower for d in safe):
        return False, 0.05
    
    return False, 0.15

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the landing page"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    print(f"[PAD.ai] GET / - Serving: {index_path}")
    
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>PAD.ai - index.html not found</h1>")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "PAD.ai"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    is_phishing, confidence = detect_phishing(request.url, request.email_content)
    return PredictionResponse(
        is_phishing=is_phishing,
        confidence=confidence,
        features_analyzed={
            "url_length": len(request.url),
            "has_email_content": bool(request.email_content)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
