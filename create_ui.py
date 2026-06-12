"""Generate Dynamic Premium PAD.ai UI with animations"""

html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PAD.ai - Security Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --primary: #6366f1;
  --primary-dark: #4f46e5;
  --danger: #ef4444;
  --success: #10b981;
  --bg-dark: #0f0f1a;
  --bg-card: rgba(255,255,255,0.03);
  --border: rgba(255,255,255,0.1);
  --text: #fff;
  --text-muted: #94a3b8;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family: 'Inter', sans-serif;
  background: var(--bg-dark);
  color: var(--text);
  min-height: 100vh;
  overflow-x: hidden;
}

/* Animated Background */
.bg-effects {
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}
.orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(100px);
  opacity: 0.5;
  animation: float 20s ease-in-out infinite;
}
.orb-1 { width: 600px; height: 600px; background: #6366f1; top: -200px; right: -200px; }
.orb-2 { width: 400px; height: 400px; background: #ec4899; bottom: -100px; left: -100px; animation-delay: -10s; }
.orb-3 { width: 300px; height: 300px; background: #06b6d4; top: 50%; left: 50%; animation-delay: -5s; }
@keyframes float {
  0%, 100% { transform: translate(0, 0) scale(1); }
  25% { transform: translate(50px, -50px) scale(1.1); }
  50% { transform: translate(-30px, 30px) scale(0.95); }
  75% { transform: translate(-50px, -30px) scale(1.05); }
}

/* Grid Pattern */
.grid-pattern {
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  background-image: linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px);
  background-size: 60px 60px;
  pointer-events: none;
}

/* Header */
.header {
  position: sticky;
  top: 0;
  background: rgba(15,15,26,0.8);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  z-index: 100;
}
.logo {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.logo-icon {
  width: 42px; height: 42px;
  background: linear-gradient(135deg, #6366f1, #ec4899);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.25rem;
  box-shadow: 0 0 30px rgba(99,102,241,0.4);
  animation: glow 3s ease-in-out infinite;
}
@keyframes glow {
  0%, 100% { box-shadow: 0 0 20px rgba(99,102,241,0.4); }
  50% { box-shadow: 0 0 40px rgba(99,102,241,0.6); }
}
.logo-text {
  font-size: 1.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #fff, #94a3b8);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.nav { display: flex; gap: 1rem; }
.nav a {
  text-decoration: none;
  color: var(--text-muted);
  padding: 0.75rem 1.25rem;
  border-radius: 10px;
  font-weight: 500;
  transition: all 0.3s;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.nav a:hover, .nav a.active {
  background: rgba(99,102,241,0.2);
  color: #fff;
}
.status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--success);
  font-weight: 500;
}
.status-dot {
  width: 10px; height: 10px;
  background: var(--success);
  border-radius: 50%;
  animation: pulse 2s infinite;
  box-shadow: 0 0 10px var(--success);
}
@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.2); opacity: 0.7; }
}

/* Content */
.content { position: relative; z-index: 1; padding: 3rem 2rem; max-width: 1000px; margin: 0 auto; }
.title {
  font-size: 3rem;
  font-weight: 800;
  text-align: center;
  margin-bottom: 1rem;
  background: linear-gradient(135deg, #fff 0%, #6366f1 50%, #ec4899 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shimmer 3s linear infinite;
  background-size: 200% 100%;
}
@keyframes shimmer {
  0% { background-position: 100% 0; }
  100% { background-position: -100% 0; }
}
.subtitle { text-align: center; color: var(--text-muted); font-size: 1.125rem; margin-bottom: 3rem; }

/* Card */
.card {
  background: var(--bg-card);
  backdrop-filter: blur(20px);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 2rem;
  margin-bottom: 2rem;
  transition: all 0.3s;
}
.card:hover { border-color: rgba(99,102,241,0.3); box-shadow: 0 0 40px rgba(99,102,241,0.1); }

.textarea {
  width: 100%;
  min-height: 140px;
  background: rgba(0,0,0,0.3);
  border: 2px solid var(--border);
  border-radius: 16px;
  padding: 1.25rem;
  font-family: inherit;
  font-size: 1rem;
  color: #fff;
  resize: vertical;
  transition: all 0.3s;
}
.textarea:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 20px rgba(99,102,241,0.2); }
.textarea::placeholder { color: #64748b; }

.btn {
  width: 100%;
  background: linear-gradient(135deg, #6366f1, #ec4899);
  color: #fff;
  border: none;
  padding: 1.25rem;
  border-radius: 16px;
  font-size: 1.125rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  margin-top: 1.5rem;
  transition: all 0.3s;
  position: relative;
  overflow: hidden;
}
.btn::before {
  content: '';
  position: absolute;
  top: 0; left: -100%;
  width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
  transition: 0.5s;
}
.btn:hover::before { left: 100%; }
.btn:hover { transform: translateY(-3px); box-shadow: 0 10px 40px rgba(99,102,241,0.4); }
.btn:disabled { opacity: 0.7; cursor: not-allowed; transform: none; }

/* Results */
.result { display: none; animation: slideUp 0.5s cubic-bezier(0.4, 0, 0.2, 1); }
.result.show { display: block; }
@keyframes slideUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }

.result-grid { display: grid; grid-template-columns: 1fr 280px; gap: 2rem; }
@media(max-width:800px) { .result-grid { grid-template-columns: 1fr; } }

.result-header { display: flex; gap: 1rem; align-items: flex-start; margin-bottom: 1.5rem; }
.result-icon {
  width: 60px; height: 60px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.75rem;
  animation: pop 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}
@keyframes pop { 0% { transform: scale(0); } 100% { transform: scale(1); } }
.result-icon.danger { background: rgba(239,68,68,0.2); box-shadow: 0 0 30px rgba(239,68,68,0.3); }
.result-icon.safe { background: rgba(16,185,129,0.2); box-shadow: 0 0 30px rgba(16,185,129,0.3); }
.result-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem; }
.result-title.danger { color: var(--danger); }
.result-title.safe { color: var(--success); }
.result-score { color: var(--text-muted); }
.result-desc { color: #cbd5e1; line-height: 1.7; margin-bottom: 1.5rem; }

.risk-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem; }
.risk-tags { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.risk-tag {
  background: rgba(99,102,241,0.2);
  border: 1px solid rgba(99,102,241,0.3);
  padding: 0.5rem 1rem;
  border-radius: 25px;
  font-size: 0.875rem;
  color: #c7d2fe;
  animation: fadeIn 0.3s ease forwards;
  opacity: 0;
}
.risk-tag:nth-child(1) { animation-delay: 0.1s; }
.risk-tag:nth-child(2) { animation-delay: 0.2s; }
.risk-tag:nth-child(3) { animation-delay: 0.3s; }
.risk-tag:nth-child(4) { animation-delay: 0.4s; }
@keyframes fadeIn { to { opacity: 1; } }

.features {
  background: rgba(0,0,0,0.3);
  border-radius: 16px;
  padding: 1.5rem;
  border: 1px solid var(--border);
}
.features-title { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; }
.feature-row {
  display: flex;
  justify-content: space-between;
  padding: 0.875rem 0;
  border-bottom: 1px solid var(--border);
  animation: slideIn 0.3s ease forwards;
  opacity: 0;
  transform: translateX(-10px);
}
.feature-row:last-child { border-bottom: none; }
.feature-row:nth-child(1) { animation-delay: 0.1s; }
.feature-row:nth-child(2) { animation-delay: 0.15s; }
.feature-row:nth-child(3) { animation-delay: 0.2s; }
.feature-row:nth-child(4) { animation-delay: 0.25s; }
@keyframes slideIn { to { opacity: 1; transform: translateX(0); } }
.feature-name { color: var(--text-muted); }
.feature-value { font-weight: 600; color: #fff; }
.feature-value.danger { color: var(--danger); }
.feature-value.success { color: var(--success); }

.footer { text-align: center; padding: 3rem 2rem; color: var(--text-muted); font-size: 0.875rem; position: relative; z-index: 1; }
.footer a { color: var(--primary); text-decoration: none; }

/* Loading Animation */
.loading { display: inline-block; }
.loading::after {
  content: '';
  display: inline-block;
  width: 20px; height: 20px;
  border: 2px solid rgba(255,255,255,0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="bg-effects">
  <div class="orb orb-1"></div>
  <div class="orb orb-2"></div>
  <div class="orb orb-3"></div>
</div>
<div class="grid-pattern"></div>

<header class="header">
  <div class="logo">
    <div class="logo-icon">🛡️</div>
    <span class="logo-text">PAD.ai</span>
  </div>
  <nav class="nav">
    <a href="#" class="active">🔍 Detection Center</a>
    <a href="/docs">📊 API Docs</a>
  </nav>
  <div class="status"><div class="status-dot"></div>System Live</div>
</header>

<div class="content">
  <h1 class="title">Detection Center</h1>
  <p class="subtitle">Analyze URLs or email content to identify potential phishing threats using our hybrid AI engine.</p>
  
  <div class="card">
    <textarea class="textarea" id="input" placeholder="Paste a URL or email message here to analyze..."></textarea>
    <button class="btn" id="scanBtn" onclick="scan()">
      <span id="btnIcon">⚡</span>
      <span id="btnText">Run Security Audit</span>
    </button>
  </div>
  
  <div class="result card" id="result">
    <div class="result-grid">
      <div>
        <div class="result-header">
          <div class="result-icon" id="resultIcon">⚠️</div>
          <div>
            <div class="result-title" id="resultTitle">PHISHING DETECTED</div>
            <div class="result-score" id="resultScore">Confidence: 95.0%</div>
          </div>
        </div>
        <p class="result-desc" id="resultDesc"></p>
        <div class="risk-factors">
          <div class="risk-label">Identified Risk Factors</div>
          <div class="risk-tags" id="riskTags"></div>
        </div>
      </div>
      <div class="features">
        <div class="features-title">Feature Vector Output</div>
        <div id="featureList"></div>
      </div>
    </div>
  </div>
</div>

<footer class="footer">
  © 2026 PAD.ai Phishing Detection System<br>
  Domain: Cyber Security | Developed with FastAPI & AI
</footer>

<script>
async function scan() {
  const input = document.getElementById('input').value;
  if (!input.trim()) { alert('Please enter a URL or message to analyze'); return; }
  
  const btn = document.getElementById('scanBtn');
  const btnIcon = document.getElementById('btnIcon');
  const btnText = document.getElementById('btnText');
  
  btnIcon.innerHTML = '';
  btnIcon.className = 'loading';
  btnText.textContent = 'Analyzing...';
  btn.disabled = true;
  
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: input, email_content: input })
    });
    const data = await res.json();
    
    const isPhishing = data.is_phishing;
    const confidence = (data.confidence * 100).toFixed(1);
    
    document.getElementById('resultIcon').className = 'result-icon ' + (isPhishing ? 'danger' : 'safe');
    document.getElementById('resultIcon').textContent = isPhishing ? '⚠️' : '✅';
    document.getElementById('resultTitle').className = 'result-title ' + (isPhishing ? 'danger' : 'safe');
    document.getElementById('resultTitle').textContent = isPhishing ? 'PHISHING DETECTED' : 'NO THREATS FOUND';
    document.getElementById('resultScore').textContent = 'Confidence: ' + confidence + '%';
    
    document.getElementById('resultDesc').textContent = isPhishing 
      ? 'This content shows characteristics commonly associated with phishing attempts. The AI has identified suspicious patterns that may indicate malicious intent. Exercise extreme caution.'
      : 'The analyzed content appears to be legitimate and safe. No malicious patterns or suspicious indicators were detected by our security engine.';
    
    const risks = [];
    const inputLower = input.toLowerCase();
    if (inputLower.includes('phishing') || inputLower.includes('scam')) risks.push('Suspicious Keywords');
    if (inputLower.includes('urgent') || inputLower.includes('immediately')) risks.push('Urgency Language');
    if (inputLower.includes('verify') || inputLower.includes('confirm')) risks.push('Verification Request');
    if (inputLower.includes('password') || inputLower.includes('credit')) risks.push('Sensitive Data Request');
    if (input.length > 80) risks.push('Unusual URL Length');
    if (!input.startsWith('https://')) risks.push('Missing HTTPS');
    
    document.getElementById('riskTags').innerHTML = risks.length 
      ? risks.map(r => '<span class="risk-tag">' + r + '</span>').join('')
      : '<span class="risk-tag">None Detected</span>';
    
    const len = data.features_analyzed?.len || input.length;
    const dots = (input.match(/\\./g) || []).length;
    const hasAt = input.includes('@');
    const isHttps = input.startsWith('https://');
    
    document.getElementById('featureList').innerHTML = 
      '<div class="feature-row"><span class="feature-name">Input Length</span><span class="feature-value">' + len + ' chars</span></div>' +
      '<div class="feature-row"><span class="feature-name">Contains @</span><span class="feature-value">' + (hasAt ? 'Yes' : 'No') + '</span></div>' +
      '<div class="feature-row"><span class="feature-name">HTTPS</span><span class="feature-value ' + (isHttps ? 'success' : 'danger') + '">' + (isHttps ? 'Present' : 'Missing') + '</span></div>' +
      '<div class="feature-row"><span class="feature-name">Domain Dots</span><span class="feature-value">' + dots + '</span></div>' +
      '<div class="feature-row"><span class="feature-name">Risk Score</span><span class="feature-value ' + (isPhishing ? 'danger' : 'success') + '">' + confidence + '%</span></div>';
    
    document.getElementById('result').classList.add('show');
    document.getElementById('result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    alert('Error connecting to API: ' + err.message);
  } finally {
    btnIcon.className = '';
    btnIcon.textContent = '⚡';
    btnText.textContent = 'Run Security Audit';
    btn.disabled = false;
  }
}

document.getElementById('input').addEventListener('keydown', function(e) {
  if (e.ctrlKey && e.key === 'Enter') scan();
});
</script>
</body>
</html>'''

with open('static/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('✨ Dynamic Premium UI created!')
print('Features:')
print('  - Animated gradient orbs background')
print('  - Grid pattern overlay')
print('  - Glassmorphism cards')
print('  - Shimmer title animation')
print('  - Glowing logo effect')
print('  - Smooth scroll animations')
print('  - Risk factor fade-in')
print('  - Feature vector slide-in')
print('  - Button shine effect')
print('')
print('Restart server: uvicorn src.main:app --host 0.0.0.0 --port 8000')
