"""Setup script to create PAD.ai files in correct location"""
import os

# Create static folder
os.makedirs('static', exist_ok=True)

# Create index.html
html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>PAD.ai</title>
<style>
body{font-family:Arial;background:#0a0a0f;color:#fff;display:flex;justify-content:center;align-items:center;height:100vh;margin:0}
.card{background:#1a1a25;padding:40px;border-radius:20px;text-align:center;max-width:500px}
h1{color:#00d4ff}
input{width:100%;padding:15px;margin:20px 0;border:2px solid #333;background:#12121a;color:#fff;border-radius:10px;font-size:16px}
button{background:linear-gradient(135deg,#00d4ff,#7c3aed);border:none;padding:15px 40px;color:#fff;border-radius:10px;font-size:16px;cursor:pointer}
button:hover{transform:scale(1.05)}
#result{margin-top:20px;padding:20px;border-radius:10px}
.safe{background:#10b981}
.danger{background:#ef4444}
</style>
</head>
<body>
<div class="card">
<h1>PAD.ai</h1>
<p>Phishing Attack Detection</p>
<input type="text" id="url" placeholder="Enter URL to scan...">
<button id="btn">Scan URL</button>
<div id="result"></div>
</div>
<script>
document.getElementById('btn').onclick = async function() {
  const url = document.getElementById('url').value;
  const res = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({url: url})
  });
  const data = await res.json();
  const el = document.getElementById('result');
  el.className = data.is_phishing ? 'danger' : 'safe';
  el.innerHTML = data.is_phishing ? 'DANGER: Phishing Detected!' : 'SAFE: No Threats Found';
};
</script>
</body>
</html>"""

with open('static/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print("Created static/index.html")

# Update main.py
main_py = '''"""PAD.ai API"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import os

print("PAD.ai Starting...")

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static")
print(f"Static: {STATIC}, exists: {os.path.exists(STATIC)}")

app = FastAPI(title="PAD.ai")

if os.path.exists(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

class Req(BaseModel):
    url: str
    email_content: Optional[str] = None

class Res(BaseModel):
    is_phishing: bool
    confidence: float
    features_analyzed: dict

@app.get("/", response_class=HTMLResponse)
async def home():
    p = os.path.join(STATIC, "index.html")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>PAD.ai</h1>")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=Res)
def predict(req: Req):
    danger = any(x in req.url.lower() for x in ["phishing","malware","suspicious"])
    return Res(is_phishing=danger, confidence=0.95 if danger else 0.05, features_analyzed={"len": len(req.url)})
'''

with open('src/main.py', 'w', encoding='utf-8') as f:
    f.write(main_py)
print("Updated src/main.py")

# Delete pycache
import shutil
try:
    shutil.rmtree('src/__pycache__')
    print("Deleted cache")
except:
    pass

print("\nDone! Now run: uvicorn src.main:app --reload")
