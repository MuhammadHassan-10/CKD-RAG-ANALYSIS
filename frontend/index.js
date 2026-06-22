/* ═══════════════════════════════════════════════════════════════════════════
   KDIGO CKD Dual RAG Assistant — Frontend Logic
   ═══════════════════════════════════════════════════════════════════════════ */

const API_BASE = window.location.origin;
const SESSION_ID = crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36);

// ── DOM elements ──────────────────────────────────────────────────────────
const chatMessages   = document.getElementById('chat-messages');
const chatInput      = document.getElementById('chat-input');
const btnSend        = document.getElementById('btn-send');
const btnIngest      = document.getElementById('btn-ingest');
const nodeCount      = document.getElementById('node-count');
const relCount       = document.getElementById('rel-count');
const contextPanel   = document.getElementById('context-panel');
const panelContent   = document.getElementById('panel-content');
const btnClosePanel  = document.getElementById('btn-close-panel');
const ingestOverlay  = document.getElementById('ingest-overlay');
const progressBar    = document.getElementById('progress-bar');
const progressText   = document.getElementById('progress-text');
const statChunks     = document.getElementById('stat-chunks');
const statEntities   = document.getElementById('stat-entities');
const statRelations  = document.getElementById('stat-relations');
const compareInput   = document.getElementById('compare-input');
const btnCompare     = document.getElementById('btn-compare');
const compareResults = document.getElementById('compare-results');
const dashContent    = document.getElementById('dashboard-content');

// ── State ─────────────────────────────────────────────────────────────────
let isLoading = false;
let selectedRagType = 'agentic';
let comparisonHistory = [];

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    fetchGraphStats();
    setupInputHandlers();
    setupTabs();
    setupRagSelector();
    loadComparisonHistory();
});

// ── Tab navigation ────────────────────────────────────────────────────────
function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            // Update buttons
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            // Update panels
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            document.getElementById(`panel-${tab}`).classList.add('active');
            // Refresh dashboard when switching to it
            if (tab === 'dashboard') renderDashboard();
        });
    });
}

// ── RAG selector ──────────────────────────────────────────────────────────
function setupRagSelector() {
    document.querySelectorAll('.rag-option').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.rag-option').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedRagType = btn.dataset.rag;
        });
    });
}

function setupInputHandlers() {
    // Textarea auto-resize for chat
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + 'px';
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    btnSend.addEventListener('click', sendMessage);
    btnIngest.addEventListener('click', startIngestion);
    btnClosePanel.addEventListener('click', () => contextPanel.classList.remove('open'));

    // Compare input
    compareInput.addEventListener('input', () => {
        compareInput.style.height = 'auto';
        compareInput.style.height = Math.min(compareInput.scrollHeight, 140) + 'px';
    });

    compareInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            runComparison();
        }
    });

    btnCompare.addEventListener('click', runComparison);
}

// ── Fetch graph stats ─────────────────────────────────────────────────────
async function fetchGraphStats() {
    try {
        const res = await fetch(`${API_BASE}/api/graph-stats`);
        if (res.ok) {
            const data = await res.json();
            nodeCount.textContent = formatNumber(data.total_nodes);
            relCount.textContent = formatNumber(data.total_relationships);
        }
    } catch (err) {
        console.warn('Could not fetch graph stats:', err);
    }
}

function formatNumber(n) {
    if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
    return n.toString();
}

// ── Send message ──────────────────────────────────────────────────────────
async function sendMessage() {
    const question = chatInput.value.trim();
    if (!question || isLoading) return;

    isLoading = true;
    btnSend.disabled = true;

    const welcome = document.getElementById('welcome-msg');
    if (welcome) welcome.style.display = 'none';

    appendMessage('user', question);
    chatInput.value = '';
    chatInput.style.height = 'auto';

    const typingId = showTypingIndicator();

    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                session_id: SESSION_ID,
                rag_type: selectedRagType,
            }),
        });

        removeTypingIndicator(typingId);

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
            appendMessage('assistant', `⚠️ Error: ${err.detail || res.statusText}`, null, null, null);
        } else {
            const data = await res.json();
            appendMessage('assistant', data.answer, data.graph_context, data.sources, data);
        }
    } catch (err) {
        removeTypingIndicator(typingId);
        appendMessage('assistant', `⚠️ Network error: ${err.message}`, null, null, null);
    }

    isLoading = false;
    btnSend.disabled = false;
    chatInput.focus();
    fetchGraphStats();
}

function askSuggestion(btn) {
    chatInput.value = btn.textContent;
    sendMessage();
}

// ── Append message to chat ────────────────────────────────────────────────
function appendMessage(role, text, graphContext, sources, meta) {
    const msg = document.createElement('div');
    msg.className = `message ${role}-message`;

    const avatarSVG = role === 'assistant'
        ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="12" cy="12" r="3"/><circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/>
            <circle cx="5" cy="18" r="2"/><circle cx="19" cy="18" r="2"/>
            <line x1="7" y1="6" x2="10" y2="10"/><line x1="17" y1="6" x2="14" y2="10"/>
            <line x1="7" y1="18" x2="10" y2="14"/><line x1="17" y1="18" x2="14" y2="14"/></svg>`
        : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/>
            <circle cx="12" cy="7" r="4"/></svg>`;

    const ragBadge = (role === 'assistant' && meta && meta.rag_type)
        ? `<span class="rag-badge rag-badge-${meta.rag_type}">${meta.rag_type}</span>` : '';

    const timeBadge = (role === 'assistant' && meta && meta.response_time_ms)
        ? `<span class="time-badge">${(meta.response_time_ms / 1000).toFixed(1)}s</span>` : '';

    let contentHtml = `
        <div class="message-avatar">${avatarSVG}</div>
        <div class="message-content">
            <div class="message-header">${role === 'assistant' ? 'RAG Assistant' : 'You'} ${ragBadge} ${timeBadge}</div>
            <div class="message-text">${role === 'assistant' ? renderMarkdown(text) : escapeHtml(text)}</div>`;

    // Steps trace for agentic
    if (role === 'assistant' && meta && meta.steps_taken && meta.steps_taken.length > 0) {
        contentHtml += `<div class="steps-trace">
            <span class="steps-label">Agent steps:</span>
            ${meta.steps_taken.map(s => `<span class="step-chip">${escapeHtml(s)}</span>`).join('<span class="step-arrow">→</span>')}
        </div>`;
    }

    // Show context button
    if (role === 'assistant' && graphContext && graphContext.length > 0) {
        contentHtml += `
            <button class="btn-show-context" onclick="showContext(this)"
                    data-context='${escapeAttr(JSON.stringify({ triples: graphContext, sources: sources || [] }))}'>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="3"/><circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/>
                    <line x1="7" y1="6" x2="10" y2="10"/><line x1="17" y1="6" x2="14" y2="10"/></svg>
                View Graph Context (${graphContext.length} triples)
            </button>`;
    }

    contentHtml += '</div>';
    msg.innerHTML = contentHtml;
    chatMessages.appendChild(msg);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ── Show context panel ────────────────────────────────────────────────────
function showContext(btn) {
    const data = JSON.parse(btn.dataset.context);
    renderContextPanel(data.triples, data.sources);
    contextPanel.classList.add('open');
}

function renderContextPanel(triples, sources) {
    let html = '';
    if (triples && triples.length > 0) {
        html += `<div class="context-section">
            <div class="context-section-title">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="3"/><circle cx="5" cy="6" r="2"/><line x1="7" y1="6" x2="10" y2="10"/></svg>
                Knowledge Graph Triples (${triples.length})
            </div>`;
        for (const t of triples) {
            html += `<div class="triple-item">
                <span class="triple-entity">${escapeHtml(t.source)}</span>
                <span class="triple-arrow">→</span>
                <span class="triple-rel">${escapeHtml(t.relationship)}</span>
                <span class="triple-arrow">→</span>
                <span class="triple-entity">${escapeHtml(t.target)}</span>
            </div>`;
        }
        html += '</div>';
    }
    if (sources && sources.length > 0) {
        html += `<div class="context-section">
            <div class="context-section-title">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/></svg>
                Source Chunks (${sources.length})
            </div>`;
        for (const s of sources) {
            html += `<div class="source-item">
                <div class="source-meta">Page ${s.page || '?'}${s.section ? ' · ' + escapeHtml(s.section) : ''}${s.score ? ' · Score: ' + s.score.toFixed(3) : ''}</div>
                ${escapeHtml(s.text)}
            </div>`;
        }
        html += '</div>';
    }
    if (!html) html = '<div class="panel-empty"><p>No graph context available.</p></div>';
    panelContent.innerHTML = html;
}

// ── Typing indicator ──────────────────────────────────────────────────────
function showTypingIndicator() {
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.className = 'message assistant-message';
    div.id = id;
    div.innerHTML = `
        <div class="message-avatar">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <circle cx="12" cy="12" r="3"/><circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/>
                <circle cx="5" cy="18" r="2"/><circle cx="19" cy="18" r="2"/>
                <line x1="7" y1="6" x2="10" y2="10"/><line x1="17" y1="6" x2="14" y2="10"/>
                <line x1="7" y1="18" x2="10" y2="14"/><line x1="17" y1="18" x2="14" y2="14"/></svg>
        </div>
        <div class="message-content">
            <div class="message-header">RAG Assistant</div>
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ═══════════════════════════════════════════════════════════════════════════
//  COMPARE
// ═══════════════════════════════════════════════════════════════════════════

async function runComparison() {
    const question = compareInput.value.trim();
    if (!question || isLoading) return;

    isLoading = true;
    btnCompare.disabled = true;
    btnCompare.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin">
        <line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/>
        <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/>
        <line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/></svg> Running…`;

    try {
        const res = await fetch(`${API_BASE}/api/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, session_id: SESSION_ID }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
            compareResults.innerHTML = `<div class="compare-error">⚠️ ${err.detail || res.statusText}</div>`;
        } else {
            const data = await res.json();
            comparisonHistory.push(data);
            renderComparisonResult(data);
        }
    } catch (err) {
        compareResults.innerHTML = `<div class="compare-error">⚠️ Network error: ${err.message}</div>`;
    }

    isLoading = false;
    btnCompare.disabled = false;
    btnCompare.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg> Compare`;
}

function renderComparisonResult(data) {
    const { traditional, agentic, traditional_metrics: tm, agentic_metrics: am, winner, summary } = data;

    const winnerLabel = winner === 'traditional' ? 'Traditional RAG' : winner === 'agentic' ? 'Agentic RAG' : 'Tie';
    const winnerClass = winner === 'tie' ? 'winner-tie' : `winner-${winner}`;

    let html = `
    <div class="compare-result-card">
        <div class="compare-question">"${escapeHtml(data.question)}"</div>

        <!-- Winner banner -->
        <div class="winner-banner ${winnerClass}">
            <span class="winner-icon">${winner === 'tie' ? '🤝' : '🏆'}</span>
            <span class="winner-text">${winnerLabel}</span>
            <span class="winner-summary">${escapeHtml(summary)}</span>
        </div>

        <!-- Metrics comparison -->
        <div class="metrics-grid">
            ${renderMetricBar('Faithfulness', tm.faithfulness, am.faithfulness)}
            ${renderMetricBar('Answer Relevancy', tm.answer_relevancy, am.answer_relevancy)}
            ${renderMetricBar('Context Precision', tm.context_precision, am.context_precision)}
            ${renderMetricBar('Context Recall', tm.context_recall, am.context_recall)}
            ${renderMetricBar('Token Efficiency', tm.token_efficiency, am.token_efficiency)}
            ${renderMetricBar('Response Time', 1 - Math.min(tm.response_time_ms / 30000, 1), 1 - Math.min(am.response_time_ms / 30000, 1), `${(tm.response_time_ms/1000).toFixed(1)}s`, `${(am.response_time_ms/1000).toFixed(1)}s`)}
        </div>

        <!-- Radar chart -->
        <div class="radar-section">
            <h3 class="section-title">Metric Radar</h3>
            <canvas id="radar-canvas-${Date.now()}" width="360" height="360" class="radar-canvas"></canvas>
        </div>

        <!-- Side-by-side answers -->
        <div class="answers-grid">
            <div class="answer-card traditional-card">
                <div class="answer-header">
                    <span class="rag-dot traditional-dot"></span>
                    <h3>Traditional RAG</h3>
                    <span class="time-badge">${(traditional.response_time_ms / 1000).toFixed(1)}s</span>
                </div>
                <div class="answer-steps">
                    ${traditional.steps_taken.map(s => `<span class="step-chip-sm">${escapeHtml(s)}</span>`).join(' ')}
                </div>
                <div class="answer-body">${renderMarkdown(traditional.answer)}</div>
                <div class="answer-meta">${traditional.sources.length} sources</div>
            </div>
            <div class="answer-card agentic-card">
                <div class="answer-header">
                    <span class="rag-dot agentic-dot"></span>
                    <h3>Agentic RAG</h3>
                    <span class="time-badge">${(agentic.response_time_ms / 1000).toFixed(1)}s</span>
                </div>
                <div class="answer-steps">
                    ${agentic.steps_taken.map(s => `<span class="step-chip-sm">${escapeHtml(s)}</span>`).join('<span class="step-arrow-sm">→</span>')}
                </div>
                <div class="answer-body">${renderMarkdown(agentic.answer)}</div>
                <div class="answer-meta">${agentic.sources.length} sources · ${agentic.graph_context.length} triples</div>
            </div>
        </div>
    </div>`;

    compareResults.innerHTML = html;

    // Draw radar chart
    const canvasId = compareResults.querySelector('.radar-canvas').id;
    drawRadarChart(canvasId, tm, am);
}

function renderMetricBar(label, tradVal, agentVal, tradLabel, agentLabel) {
    const tPct = Math.round(tradVal * 100);
    const aPct = Math.round(agentVal * 100);
    const tDisplay = tradLabel || `${tPct}%`;
    const aDisplay = agentLabel || `${aPct}%`;
    const tradWin = tradVal > agentVal + 0.02;
    const agentWin = agentVal > tradVal + 0.02;

    return `<div class="metric-row">
        <div class="metric-label">${label}</div>
        <div class="metric-bars">
            <div class="metric-bar-row">
                <span class="metric-tag traditional-tag ${tradWin ? 'metric-winner' : ''}">T</span>
                <div class="metric-bar-track"><div class="metric-bar-fill traditional-fill" style="width:${tPct}%"></div></div>
                <span class="metric-value">${tDisplay}</span>
            </div>
            <div class="metric-bar-row">
                <span class="metric-tag agentic-tag ${agentWin ? 'metric-winner' : ''}">A</span>
                <div class="metric-bar-track"><div class="metric-bar-fill agentic-fill" style="width:${aPct}%"></div></div>
                <span class="metric-value">${aDisplay}</span>
            </div>
        </div>
    </div>`;
}

// ── Radar chart ───────────────────────────────────────────────────────────
function drawRadarChart(canvasId, tm, am) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const cx = 180, cy = 180, r = 130;
    const labels = ['Faithfulness', 'Relevancy', 'Precision', 'Recall', 'Efficiency'];
    const tradVals = [tm.faithfulness, tm.answer_relevancy, tm.context_precision, tm.context_recall, tm.token_efficiency];
    const agentVals = [am.faithfulness, am.answer_relevancy, am.context_precision, am.context_recall, am.token_efficiency];
    const n = labels.length;
    const angleStep = (Math.PI * 2) / n;
    const startAngle = -Math.PI / 2;

    // Clear
    ctx.clearRect(0, 0, 360, 360);

    // Grid rings
    for (let ring = 1; ring <= 5; ring++) {
        const rr = (ring / 5) * r;
        ctx.beginPath();
        for (let i = 0; i <= n; i++) {
            const a = startAngle + i * angleStep;
            const x = cx + Math.cos(a) * rr;
            const y = cy + Math.sin(a) * rr;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.12)';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Axis lines + labels
    ctx.font = '11px Inter, sans-serif';
    ctx.fillStyle = 'rgba(148, 163, 184, 0.7)';
    ctx.textAlign = 'center';
    for (let i = 0; i < n; i++) {
        const a = startAngle + i * angleStep;
        const x1 = cx + Math.cos(a) * r;
        const y1 = cy + Math.sin(a) * r;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x1, y1);
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)';
        ctx.stroke();
        // Label
        const lx = cx + Math.cos(a) * (r + 20);
        const ly = cy + Math.sin(a) * (r + 20);
        ctx.fillText(labels[i], lx, ly + 4);
    }

    // Draw polygon
    function drawPoly(values, fillColor, strokeColor) {
        ctx.beginPath();
        for (let i = 0; i <= n; i++) {
            const idx = i % n;
            const a = startAngle + idx * angleStep;
            const v = Math.max(0, Math.min(1, values[idx]));
            const x = cx + Math.cos(a) * r * v;
            const y = cy + Math.sin(a) * r * v;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Points
        for (let i = 0; i < n; i++) {
            const a = startAngle + i * angleStep;
            const v = Math.max(0, Math.min(1, values[i]));
            const x = cx + Math.cos(a) * r * v;
            const y = cy + Math.sin(a) * r * v;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = strokeColor;
            ctx.fill();
        }
    }

    drawPoly(tradVals, 'rgba(251, 146, 60, 0.15)', 'rgba(251, 146, 60, 0.8)');
    drawPoly(agentVals, 'rgba(6, 182, 212, 0.15)', 'rgba(6, 182, 212, 0.8)');

    // Legend
    ctx.font = '12px Inter, sans-serif';
    ctx.fillStyle = 'rgba(251, 146, 60, 0.9)';
    ctx.fillRect(20, 340, 12, 12);
    ctx.fillText('Traditional', 70, 351);
    ctx.fillStyle = 'rgba(6, 182, 212, 0.9)';
    ctx.fillRect(160, 340, 12, 12);
    ctx.fillText('Agentic', 202, 351);
}

// ═══════════════════════════════════════════════════════════════════════════
//  DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════

async function loadComparisonHistory() {
    try {
        const res = await fetch(`${API_BASE}/api/compare/history`);
        if (res.ok) {
            const data = await res.json();
            if (data.length > 0) comparisonHistory = data;
        }
    } catch (err) {
        console.warn('Could not load comparison history:', err);
    }
}

function renderDashboard() {
    if (comparisonHistory.length === 0) {
        dashContent.innerHTML = `<div class="dashboard-empty">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity="0.2">
                <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
                <rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
            </svg>
            <p>Run some comparisons first to see aggregated evaluation metrics here.</p>
        </div>`;
        return;
    }

    // Aggregate metrics
    const metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'token_efficiency'];
    const tradAvgs = {};
    const agentAvgs = {};
    for (const m of metrics) {
        tradAvgs[m] = comparisonHistory.reduce((s, c) => s + (c.traditional_metrics[m] || 0), 0) / comparisonHistory.length;
        agentAvgs[m] = comparisonHistory.reduce((s, c) => s + (c.agentic_metrics[m] || 0), 0) / comparisonHistory.length;
    }

    const tradAvgTime = comparisonHistory.reduce((s, c) => s + (c.traditional_metrics.response_time_ms || 0), 0) / comparisonHistory.length;
    const agentAvgTime = comparisonHistory.reduce((s, c) => s + (c.agentic_metrics.response_time_ms || 0), 0) / comparisonHistory.length;

    const wins = { traditional: 0, agentic: 0, tie: 0 };
    for (const c of comparisonHistory) wins[c.winner] = (wins[c.winner] || 0) + 1;

    let html = `
    <!-- Win/Loss/Tie summary -->
    <div class="dash-summary">
        <div class="dash-stat-card">
            <div class="dash-stat-number">${comparisonHistory.length}</div>
            <div class="dash-stat-label">Comparisons</div>
        </div>
        <div class="dash-stat-card traditional-highlight">
            <div class="dash-stat-number">${wins.traditional}</div>
            <div class="dash-stat-label">Traditional Wins</div>
        </div>
        <div class="dash-stat-card agentic-highlight">
            <div class="dash-stat-number">${wins.agentic}</div>
            <div class="dash-stat-label">Agentic Wins</div>
        </div>
        <div class="dash-stat-card">
            <div class="dash-stat-number">${wins.tie}</div>
            <div class="dash-stat-label">Ties</div>
        </div>
    </div>

    <!-- Average metrics -->
    <div class="dash-section">
        <h3 class="section-title">Average Metrics</h3>
        <div class="metrics-grid">
            ${metrics.map(m => renderMetricBar(
                m.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
                tradAvgs[m],
                agentAvgs[m]
            )).join('')}
            ${renderMetricBar('Avg Response Time',
                1 - Math.min(tradAvgTime / 30000, 1),
                1 - Math.min(agentAvgTime / 30000, 1),
                `${(tradAvgTime/1000).toFixed(1)}s`,
                `${(agentAvgTime/1000).toFixed(1)}s`
            )}
        </div>
    </div>

    <!-- Radar chart -->
    <div class="dash-section">
        <h3 class="section-title">Average Radar</h3>
        <canvas id="dash-radar-canvas" width="360" height="360" class="radar-canvas"></canvas>
    </div>

    <!-- History list -->
    <div class="dash-section">
        <h3 class="section-title">Comparison History</h3>
        <div class="history-list">
            ${comparisonHistory.map((c, i) => `
                <div class="history-item">
                    <span class="history-index">#${i + 1}</span>
                    <span class="history-question">${escapeHtml(c.question.substring(0, 80))}${c.question.length > 80 ? '…' : ''}</span>
                    <span class="history-winner winner-badge-${c.winner}">${c.winner === 'tie' ? '🤝 Tie' : c.winner === 'agentic' ? '🏆 Agentic' : '🏆 Traditional'}</span>
                </div>
            `).join('')}
        </div>
    </div>`;

    dashContent.innerHTML = html;

    // Draw radar
    drawRadarChart('dash-radar-canvas',
        { faithfulness: tradAvgs.faithfulness, answer_relevancy: tradAvgs.answer_relevancy, context_precision: tradAvgs.context_precision, context_recall: tradAvgs.context_recall, token_efficiency: tradAvgs.token_efficiency },
        { faithfulness: agentAvgs.faithfulness, answer_relevancy: agentAvgs.answer_relevancy, context_precision: agentAvgs.context_precision, context_recall: agentAvgs.context_recall, token_efficiency: agentAvgs.token_efficiency }
    );
}

// ── Ingestion ─────────────────────────────────────────────────────────────
async function startIngestion() {
    btnIngest.disabled = true;
    ingestOverlay.classList.add('active');
    progressBar.style.width = '0%';
    progressText.textContent = 'Starting ingestion pipeline…';

    try {
        const res = await fetch(`${API_BASE}/api/ingest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            progressText.textContent = `Error: ${err.detail || 'Failed to start ingestion'}`;
            setTimeout(() => {
                ingestOverlay.classList.remove('active');
                btnIngest.disabled = false;
            }, 3000);
            return;
        }

        pollIngestionStatus();
    } catch (err) {
        progressText.textContent = `Network error: ${err.message}`;
        setTimeout(() => {
            ingestOverlay.classList.remove('active');
            btnIngest.disabled = false;
        }, 3000);
    }
}

async function pollIngestionStatus() {
    const poll = async () => {
        try {
            const res = await fetch(`${API_BASE}/api/status`);
            if (res.ok) {
                const data = await res.json();
                progressBar.style.width = data.progress + '%';
                progressText.textContent = data.message || data.status;
                statChunks.textContent = data.chunks_count;
                statEntities.textContent = data.entities_count;
                statRelations.textContent = data.relationships_count;

                if (data.status === 'completed') {
                    setTimeout(() => {
                        ingestOverlay.classList.remove('active');
                        btnIngest.disabled = false;
                        fetchGraphStats();
                    }, 2000);
                    return;
                }

                if (data.status === 'failed') {
                    progressText.textContent = '❌ ' + data.message;
                    setTimeout(() => {
                        ingestOverlay.classList.remove('active');
                        btnIngest.disabled = false;
                    }, 5000);
                    return;
                }
            }
        } catch (err) {
            console.warn('Poll error:', err);
        }
        setTimeout(poll, 2000);
    };
    setTimeout(poll, 1000);
}

// ── Markdown rendering ───────────────────────────────────────────────────
function renderMarkdown(text) {
    if (!text) return '';
    let html = escapeHtml(text);

    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
        `<pre><code class="language-${lang}">${code.trim()}</code></pre>`);
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    html = html.replace(/^\s*[-*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    html = html.replace(/^\s*\d+\. (.+)$/gm, '<li>$1</li>');
    html = html.replace(/\n\n/g, '</p><p>');
    html = '<p>' + html + '</p>';
    html = html.replace(/<p>\s*<\/p>/g, '');
    html = html.replace(/<p>\s*(<h[1-3]>)/g, '$1');
    html = html.replace(/(<\/h[1-3]>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)\s*<\/p>/g, '$1');
    html = html.replace(/<p>\s*(<pre>)/g, '$1');
    html = html.replace(/(<\/pre>)\s*<\/p>/g, '$1');
    html = html.replace(/\n/g, '<br>');

    return html;
}

// ── Utilities ─────────────────────────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeAttr(text) {
    return text.replace(/'/g, '&#39;').replace(/"/g, '&quot;');
}
