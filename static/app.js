// PAD.ai - Frontend Application

const API_BASE = '';

// DOM Elements
const tabBtns = document.querySelectorAll('.tab-btn');
const urlPanel = document.getElementById('url-panel');
const emailPanel = document.getElementById('email-panel');
const urlInput = document.getElementById('url-input');
const senderInput = document.getElementById('sender-input');
const emailContentInput = document.getElementById('email-content-input');
const scanUrlBtn = document.getElementById('scan-url-btn');
const scanEmailBtn = document.getElementById('scan-email-btn');
const resultContainer = document.getElementById('result-container');
const resultIcon = document.getElementById('result-icon');
const resultTitle = document.getElementById('result-title');
const resultDetails = document.getElementById('result-details');

// Tab Switching
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const tab = btn.dataset.tab;
        if (tab === 'url') {
            urlPanel.classList.add('active');
            emailPanel.classList.remove('active');
        } else {
            emailPanel.classList.add('active');
            urlPanel.classList.remove('active');
        }

        // Hide results when switching tabs
        resultContainer.classList.remove('visible', 'safe', 'danger');
    });
});

// URL Scanning
scanUrlBtn.addEventListener('click', async () => {
    const url = urlInput.value.trim();

    if (!url) {
        showNotification('Please enter a URL to scan', 'warning');
        return;
    }

    await performScan(scanUrlBtn, { url });
});

// Email Scanning
scanEmailBtn.addEventListener('click', async () => {
    const sender = senderInput.value.trim();
    const emailContent = emailContentInput.value.trim();

    if (!emailContent) {
        showNotification('Please enter email content to analyze', 'warning');
        return;
    }

    // For email, we'll extract any URLs or use a default scan URL
    await performScan(scanEmailBtn, {
        url: sender ? `mailto:${sender}` : 'email://analysis',
        email_content: emailContent,
        sender: sender
    });
});

// Demo mode - client-side phishing detection patterns
const PHISHING_PATTERNS = [
    /phishing/i, /login.*secure/i, /verify.*account/i, /suspicious/i,
    /update.*payment/i, /confirm.*identity/i, /urgent.*action/i,
    /click.*here.*now/i, /winner/i, /prize/i, /lottery/i,
    /bit\.ly/i, /tinyurl/i, /free.*money/i, /password.*reset/i,
    /amazon.*secure/i, /paypal.*verify/i, /bank.*update/i
];

const SUSPICIOUS_URL_PATTERNS = [
    /@/, /\d{4,}/, /xn--/, /\.tk$/i, /\.ml$/i, /\.ga$/i,
    /[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}/,
    /https?:\/\/[^/]+\/[^/]+\/[^/]+\/[^/]+\/[^/]+/
];

function analyzeLocally(data) {
    const url = data.url || '';
    const email = data.email_content || '';
    const combined = url + ' ' + email;

    let phishingScore = 0;
    let matchedPatterns = [];

    // Check phishing patterns
    PHISHING_PATTERNS.forEach(pattern => {
        if (pattern.test(combined)) {
            phishingScore += 15;
            matchedPatterns.push(pattern.source);
        }
    });

    // Check suspicious URL patterns
    SUSPICIOUS_URL_PATTERNS.forEach(pattern => {
        if (pattern.test(url)) {
            phishingScore += 10;
        }
    });

    // URL length check
    if (url.length > 75) phishingScore += 10;
    if (url.length > 100) phishingScore += 15;

    // Special character density
    const specialChars = (url.match(/[%&=?#@]/g) || []).length;
    if (specialChars > 5) phishingScore += 10;

    const isPhishing = phishingScore >= 25;
    const confidence = Math.min(0.99, Math.max(0.1, phishingScore / 100 + 0.3));

    return {
        is_phishing: isPhishing,
        confidence: isPhishing ? confidence : 1 - confidence,
        features_analyzed: {
            url_length: url.length,
            has_email_content: email.length > 0,
            patterns_matched: matchedPatterns.length,
            special_chars: specialChars,
            risk_score: phishingScore
        }
    };
}

// Perform API Scan
async function performScan(button, data) {
    button.classList.add('loading');
    resultContainer.classList.remove('visible', 'safe', 'danger');

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayResult(result);

    } catch (error) {
        console.log('API not available, using demo mode');
        // Fallback to client-side demo analysis
        const result = analyzeLocally(data);
        displayResult(result);
    } finally {
        button.classList.remove('loading');
    }
}

// Display Result
function displayResult(result) {
    const isPhishing = result.is_phishing;
    const confidence = (result.confidence * 100).toFixed(1);

    // Set result type
    resultContainer.classList.remove('safe', 'danger');
    resultContainer.classList.add(isPhishing ? 'danger' : 'safe');
    resultContainer.classList.add('visible');

    // Set icon
    resultIcon.textContent = isPhishing ? '⚠️' : '✅';

    // Set title
    resultTitle.textContent = isPhishing
        ? 'Phishing Threat Detected!'
        : 'No Threats Detected';

    // Build details
    const detailsHtml = `
    <div class="detail-item">
      <div class="detail-label">Status</div>
      <div class="detail-value" style="color: ${isPhishing ? 'var(--danger)' : 'var(--success)'}">
        ${isPhishing ? 'DANGEROUS' : 'SAFE'}
      </div>
    </div>
    <div class="detail-item">
      <div class="detail-label">Confidence</div>
      <div class="detail-value">${confidence}%</div>
    </div>
    <div class="detail-item">
      <div class="detail-label">URL Length</div>
      <div class="detail-value">${result.features_analyzed?.url_length || 'N/A'}</div>
    </div>
    <div class="detail-item">
      <div class="detail-label">Email Content</div>
      <div class="detail-value">${result.features_analyzed?.has_email_content ? 'Analyzed' : 'Not Provided'}</div>
    </div>
  `;

    resultDetails.innerHTML = detailsHtml;

    // Scroll to result
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Notification (simple alert fallback)
function showNotification(message, type) {
    alert(message);
}

// Enter key support
urlInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        scanUrlBtn.click();
    }
});

// Initialize - Check API health
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            console.log('PAD.ai API is healthy');
        }
    } catch (error) {
        console.warn('API health check failed. Ensure the server is running.');
    }
}

checkHealth();
