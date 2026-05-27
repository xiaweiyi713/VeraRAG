/**
 * VeraRAG SSE Client - handles real-time pipeline updates
 */

// HTML escape to prevent XSS
function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

// Stage definitions
const STAGES = [
    { id: 'task_analysis', label: '任务分析', icon: '分析' },
    { id: 'decomposition', label: '问题分解', icon: '分解' },
    { id: 'retrieval', label: '证据检索', icon: '检索' },
    { id: 'reasoning', label: '推理', icon: '推理' },
    { id: 'verification', label: '验证', icon: '验证' },
    { id: 'repair', label: '修复', icon: '修复' }
];

let currentEventData = {
    evidence: [],
    claims: [],
    uncertainty: null,
    reasoning: null,
    verification: null,
    repair: null,
    resultId: null
};

// DOM refs (only exist on the index page)
const queryForm = document.getElementById('query-form');
const questionInput = document.getElementById('question-input');
const submitBtn = document.getElementById('submit-btn');
const pipelinePanel = document.getElementById('pipeline-panel');
const stagesContainer = document.getElementById('stages-container');
const resultPanel = document.getElementById('result-panel');
const errorPanel = document.getElementById('error-panel');
const errorText = document.getElementById('error-text');

// Guard: only run query logic on the index page
const isIndexPage = !!queryForm;

// Init stages
function initStages() {
    stagesContainer.innerHTML = '';
    STAGES.forEach(s => {
        const el = document.createElement('div');
        el.id = `stage-${s.id}`;
        el.className = 'flex items-center gap-3 text-sm stage-pending';
        el.innerHTML = `
            <span class="w-6 h-6 rounded-full border border-gray-700 flex items-center justify-center text-xs" id="icon-${s.id}">○</span>
            <span>${s.label}</span>
            <span id="detail-${s.id}" class="text-xs text-gray-600 ml-2"></span>
        `;
        stagesContainer.appendChild(el);
    });
}

function setStageActive(stageId) {
    const el = document.getElementById(`stage-${stageId}`);
    const icon = document.getElementById(`icon-${stageId}`);
    if (el) el.className = 'flex items-center gap-3 text-sm stage-active';
    if (icon) {
        icon.innerHTML = '⟳';
        icon.className = 'w-6 h-6 rounded-full border border-violet-500/50 flex items-center justify-center text-xs pulse';
    }
}

function setStageDone(stageId) {
    const el = document.getElementById(`stage-${stageId}`);
    const icon = document.getElementById(`icon-${stageId}`);
    if (el) el.className = 'flex items-center gap-3 text-sm stage-done';
    if (icon) {
        icon.innerHTML = '✓';
        icon.className = 'w-6 h-6 rounded-full border border-emerald-500/50 bg-emerald-500/10 flex items-center justify-center text-xs';
    }
}

function setStageDetail(stageId, text) {
    const el = document.getElementById(`detail-${stageId}`);
    if (el) el.textContent = text;
}

// Render functions
function renderEvidence(evidence) {
    const list = document.getElementById('evidence-list');
    if (!list) return;
    list.innerHTML = evidence.map(e => `
        <div class="bg-white/[0.02] rounded-xl p-4 border border-white/5">
            <div class="flex items-center justify-between mb-2">
                <span class="text-xs text-violet-400/80 font-medium">${esc(e.source) || 'source'}</span>
                <span class="text-xs text-gray-600">${(e.combined_score || 0).toFixed(2)}</span>
            </div>
            <p class="text-sm text-gray-300">${esc(e.title)}</p>
            ${e.text_span ? `<p class="text-xs text-gray-500 mt-1 line-clamp-2">${esc(e.text_span)}</p>` : ''}
        </div>
    `).join('');
}

function renderClaims(claims) {
    const list = document.getElementById('claims-list');
    if (!claims || claims.length === 0) {
        list.innerHTML = '<p class="text-sm text-gray-500">暂无声明信息</p>';
        return;
    }
    const statusColors = {
        'supported': 'text-emerald-400 bg-emerald-400/10',
        'refuted': 'text-red-400 bg-red-400/10',
        'not_enough_info': 'text-amber-400 bg-amber-400/10'
    };
    const statusLabels = { 'supported': '已验证', 'refuted': '已反驳', 'not_enough_info': '证据不足' };
    const statusIcons = { 'supported': '●', 'refuted': '✕', 'not_enough_info': '◐' };
    // Claim Ledger header
    let html = `<div class="mb-3 flex items-center justify-between">
        <span class="text-xs text-gray-500">声明验证总览</span>
        <span class="text-xs text-gray-600">${claims.length} 条声明</span>
    </div>`;
    html += claims.map((c, i) => {
        const statusClass = statusColors[c.verification_status] || statusColors['not_enough_info'];
        const label = statusLabels[c.verification_status] || c.verification_status;
        const icon = statusIcons[c.verification_status] || '○';
        const suppEv = (c.supporting_evidence || []).join(', ');
        const confEv = (c.conflicting_evidence || []).join(', ');
        return `
            <div class="bg-white/[0.02] rounded-xl p-4 border border-white/5 mb-3">
                <div class="flex items-start gap-3">
                    <span class="text-lg ${c.verification_status === 'supported' ? 'text-emerald-400' : c.verification_status === 'refuted' ? 'text-red-400' : 'text-amber-400'} mt-0.5">${icon}</span>
                    <div class="flex-1 min-w-0">
                        <p class="text-sm text-gray-300 mb-2">${esc(c.claim)}</p>
                        <div class="flex items-center gap-3 flex-wrap">
                            <span class="text-xs px-2 py-0.5 rounded-full ${statusClass}">${label}</span>
                            <span class="text-xs text-gray-600">置信度 ${(c.confidence || 0).toFixed(2)}</span>
                            ${suppEv ? `<span class="text-xs text-emerald-500/60">支持: ${esc(suppEv)}</span>` : ''}
                            ${confEv ? `<span class="text-xs text-red-500/60">冲突: ${esc(confEv)}</span>` : ''}
                        </div>
                    </div>
                </div>
            </div>`;
    }).join('');
    list.innerHTML = html;
}

function renderConflictGraph(data) {
    const el = document.getElementById('conflict-graph');
    if (!el) return;
    const evidence = data.evidence || [];
    const claims = data.claims || [];

    // Extract conflict edges from reasoning or verification
    let conflictEdges = [];
    // Check if there's conflict data stored
    if (data.verification && data.verification.claim_verifications) {
        const verifications = data.verification.claim_verifications;
        verifications.forEach((v, i) => {
            if (v.status === 'refuted') {
                conflictEdges.push({ from: `Claim ${i+1}`, to: '证据', type: 'refuted' });
            }
        });
    }

    if (evidence.length === 0 && conflictEdges.length === 0) {
        el.innerHTML = '<p class="text-sm text-gray-500">暂无冲突信息</p>';
        return;
    }

    // SVG Conflict Graph visualization
    const width = 600, padding = 40;
    const nodeCount = evidence.length + claims.length;
    const nodeSpacing = Math.min(120, (width - padding * 2) / Math.max(nodeCount - 1, 1));

    let svg = `<svg viewBox="0 0 ${width} ${Math.max(200, nodeCount * 30 + 80)}" class="w-full" xmlns="http://www.w3.org/2000/svg">`;

    // Evidence nodes (top row)
    const evY = 50;
    evidence.forEach((ev, i) => {
        const x = padding + i * nodeSpacing;
        const score = ev.combined_score || 0.5;
        const color = score > 0.8 ? '#6ee7b7' : score > 0.6 ? '#fbbf24' : '#6b7280';
        svg += `<circle cx="${x}" cy="${evY}" r="16" fill="${color}" opacity="0.3" stroke="${color}" stroke-width="1.5"/>`;
        svg += `<text x="${x}" y="${evY + 4}" text-anchor="middle" fill="#e5e7eb" font-size="9">${esc(ev.evidence_id || 'E' + (i+1))}</text>`;
        svg += `<text x="${x}" y="${evY + 32}" text-anchor="middle" fill="#6b7280" font-size="8">${esc((ev.title || '').substring(0, 12))}${(ev.title || '').length > 12 ? '...' : ''}</text>`;
    });

    // Claim nodes (bottom row)
    const clY = evY + 90;
    claims.forEach((cl, i) => {
        const x = padding + i * nodeSpacing;
        const st = cl.verification_status;
        const color = st === 'supported' ? '#6ee7b7' : st === 'refuted' ? '#f87171' : '#fbbf24';
        svg += `<rect x="${x-20}" y="${clY-12}" width="40" height="24" rx="4" fill="${color}" opacity="0.2" stroke="${color}" stroke-width="1.5"/>`;
        svg += `<text x="${x}" y="${clY + 4}" text-anchor="middle" fill="#e5e7eb" font-size="8">C${i+1}</text>`;
        svg += `<text x="${x}" y="${clY + 28}" text-anchor="middle" fill="#6b7280" font-size="7">${esc((cl.claim || '').substring(0, 15))}...</text>`;
    });

    // Evidence → Claim support lines
    claims.forEach((cl, ci) => {
        const clX = padding + ci * nodeSpacing;
        const suppEv = cl.supporting_evidence || [];
        const confEv = cl.conflicting_evidence || [];
        evidence.forEach((ev, ei) => {
            const evX = padding + ei * nodeSpacing;
            const evId = ev.evidence_id || `E${ei+1}`;
            let lineColor = 'transparent';
            if (suppEv.includes(evId)) lineColor = '#6ee7b7';
            else if (confEv.includes(evId)) lineColor = '#f87171';
            if (lineColor !== 'transparent') {
                svg += `<line x1="${evX}" y1="${evY + 16}" x2="${clX}" y2="${clY - 12}" stroke="${lineColor}" stroke-width="1.5" opacity="0.5"/>`;
            }
        });
    });

    // Legend
    const legendY = clY + 55;
    svg += `<line x1="30" y1="${legendY}" x2="50" y2="${legendY}" stroke="#6ee7b7" stroke-width="2"/>`;
    svg += `<text x="55" y="${legendY + 3}" fill="#6b7280" font-size="9">支持</text>`;
    svg += `<line x1="100" y1="${legendY}" x2="120" y2="${legendY}" stroke="#f87171" stroke-width="2"/>`;
    svg += `<text x="125" y="${legendY + 3}" fill="#6b7280" font-size="9">反驳</text>`;
    svg += `<line x1="170" y1="${legendY}" x2="190" y2="${legendY}" stroke="#fbbf24" stroke-width="2"/>`;
    svg += `<text x="195" y="${legendY + 3}" fill="#6b7280" font-size="9">证据不足</text>`;

    svg += '</svg>';
    el.innerHTML = svg;
}

function renderUncertainty(uncertainty) {
    const el = document.getElementById('uncertainty-detail');
    if (!uncertainty) {
        el.innerHTML = '<p class="text-sm text-gray-500">暂无不确定性信息</p>';
        return;
    }
    const dims = [
        { label: '检索不确定性', key: 'retrieval_uncertainty', weight: '25%', color: 'violet' },
        { label: '证据冲突', key: 'evidence_conflict', weight: '30%', color: 'rose' },
        { label: '推理差距', key: 'reasoning_gap', weight: '20%', color: 'amber' },
        { label: '来源可靠性', key: 'source_reliability', weight: '15%', color: 'sky' },
        { label: '验证不确定性', key: 'verification_uncertainty', weight: '10%', color: 'emerald' },
    ];
    const overall = uncertainty.overall_uncertainty || uncertainty.overall || 0;
    const barColors = { violet: '#a78bfa', rose: '#fb7185', amber: '#fbbf24', sky: '#38bdf8', emerald: '#6ee7b7' };

    // Waterfall chart: stacked horizontal bars
    let waterfallHtml = '<div class="mb-4"><svg viewBox="0 0 500 120" class="w-full" xmlns="http://www.w3.org/2000/svg">';
    let x = 0;
    const barY = 30, barH = 24, scale = 500;
    dims.forEach((d, i) => {
        const val = uncertainty[d.key] || 0;
        const w = val * scale;
        const color = barColors[d.color];
        waterfallHtml += `<rect x="${x}" y="${barY}" width="${Math.max(w, 1)}" height="${barH}" fill="${color}" opacity="0.7" rx="2"/>`;
        if (w > 15) {
            waterfallHtml += `<text x="${x + w/2}" y="${barY + barH/2 + 3}" text-anchor="middle" fill="#fff" font-size="9">${(val * 100).toFixed(0)}%</text>`;
        }
        x += w;
    });
    // Overall line
    const overallX = overall * scale;
    waterfallHtml += `<line x1="${overallX}" y1="${barY - 6}" x2="${overallX}" y2="${barY + barH + 6}" stroke="#fff" stroke-width="2" opacity="0.8"/>`;
    waterfallHtml += `<text x="${overallX}" y="${barY - 10}" text-anchor="middle" fill="#e5e7eb" font-size="9">总体 ${(overall * 100).toFixed(0)}%</text>`;
    // Dimension labels below
    x = 0;
    dims.forEach((d, i) => {
        const val = uncertainty[d.key] || 0;
        const w = val * scale;
        const midX = x + w / 2;
        if (w > 30) {
            waterfallHtml += `<text x="${midX}" y="${barY + barH + 18}" text-anchor="middle" fill="#6b7280" font-size="7">${d.label}</text>`;
        }
        x += w;
    });
    waterfallHtml += '</svg></div>';

    // Detail bars
    let detailHtml = dims.map(d => {
        const val = uncertainty[d.key] || 0;
        const pct = (val * 100).toFixed(0);
        const barColor = val > 0.5 ? 'bg-amber-500' : val > 0.3 ? 'bg-yellow-500' : 'bg-emerald-500';
        return `<div>
            <div class="flex justify-between text-xs mb-1">
                <span class="text-gray-400">${d.label} <span class="text-gray-600">(${d.weight})</span></span>
                <span class="text-gray-500">${val.toFixed(3)}</span>
            </div>
            <div class="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div class="h-full rounded-full ${barColor}" style="width: ${pct}%"></div>
            </div>
        </div>`;
    }).join('');

    el.innerHTML = `
        <div class="space-y-3">
            ${waterfallHtml}
            ${detailHtml}
            <div class="pt-3 border-t border-white/5 flex justify-between text-sm">
                <span class="text-gray-300">总体不确定性</span>
                <span class="font-medium ${overall > 0.5 ? 'text-amber-400' : 'text-emerald-400'}">${overall.toFixed(3)}</span>
            </div>
            <div class="flex justify-between text-xs text-gray-600 pt-1">
                <span>置信度</span>
                <span class="font-medium text-gray-400">${(1 - overall).toFixed(3)}</span>
            </div>
        </div>`;
}

function renderRepair(repair) {
    const el = document.getElementById('repair-detail');
    if (!el) return;
    if (!repair || !repair.original_answer) {
        el.innerHTML = '<p class="text-sm text-gray-500">本次推理无需修复</p>';
        return;
    }
    const issues = (repair.issues || []).map(iss =>
        `<div class="flex items-start gap-2 text-xs">
            <span class="text-amber-400 mt-0.5">!</span>
            <span class="text-gray-400">${esc(iss)}</span>
        </div>`
    ).join('');
    const diffHtml = buildDiffHtml(repair.original_answer, repair.repaired_answer);
    el.innerHTML = `
        <div class="space-y-4">
            <div>
                <span class="text-xs text-gray-500">修复原因</span>
                <div class="mt-1 space-y-1">${issues || '<span class="text-xs text-gray-600">无具体说明</span>'}</div>
            </div>
            <div>
                <span class="text-xs text-gray-500">原始回答</span>
                <div class="mt-1 p-3 rounded-lg bg-red-500/5 border border-red-500/10 text-sm text-gray-400 line-through decoration-red-400/40">${esc(repair.original_answer)}</div>
            </div>
            <div>
                <span class="text-xs text-gray-500">修复后回答</span>
                <div class="mt-1 p-3 rounded-lg bg-emerald-500/5 border border-emerald-500/10 text-sm text-gray-300">${esc(repair.repaired_answer)}</div>
            </div>
            ${repair.confidence_delta !== undefined ? `
            <div class="flex items-center justify-between text-xs pt-2 border-t border-white/5">
                <span class="text-gray-500">置信度变化</span>
                <span class="${repair.confidence_delta >= 0 ? 'text-emerald-400' : 'text-amber-400'}">${repair.confidence_delta >= 0 ? '+' : ''}${repair.confidence_delta.toFixed(2)}</span>
            </div>` : ''}
        </div>`;
}

function buildDiffHtml(oldText, newText) {
    if (!oldText || !newText) return '';
    const oldWords = oldText.split(/(?<=[，。！？；：\s])/);
    const newWords = newText.split(/(?<=[，。！？；：\s])/);
    let html = '';
    let oi = 0, ni = 0;
    while (oi < oldWords.length || ni < newWords.length) {
        if (oi < oldWords.length && ni < newWords.length && oldWords[oi] === newWords[ni]) {
            html += esc(oldWords[oi]);
            oi++; ni++;
        } else {
            let matchedOld = false;
            for (let look = ni + 1; look < Math.min(ni + 5, newWords.length); look++) {
                if (oi < oldWords.length && newWords[look] === oldWords[oi]) {
                    while (ni < look) { html += `<span class="bg-emerald-500/20 text-emerald-300">${esc(newWords[ni])}</span>`; ni++; }
                    matchedOld = true; break;
                }
            }
            if (!matchedOld) {
                if (oi < oldWords.length) { html += `<span class="bg-red-500/15 text-red-300/60 line-through">${esc(oldWords[oi])}</span>`; oi++; }
                if (ni < newWords.length) { html += `<span class="bg-emerald-500/20 text-emerald-300">${esc(newWords[ni])}</span>`; ni++; }
            }
        }
    }
    return html;
}

// Tab switching
if (isIndexPage) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => {
                b.classList.remove('active');
                b.classList.add('text-gray-500');
            });
            btn.classList.add('active');
            btn.classList.remove('text-gray-500');
            document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
            document.getElementById(`tab-${btn.dataset.tab}`).classList.remove('hidden');
        });
    });
}

// Submit query
if (isIndexPage) {
queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    // Reset UI
    submitBtn.disabled = true;
    submitBtn.textContent = '推理中...';
    resultPanel.classList.add('hidden');
    errorPanel.classList.add('hidden');
    pipelinePanel.classList.remove('hidden');
    initStages();
    currentEventData = { evidence: [], claims: [], uncertainty: null, reasoning: null, verification: null, repair: null, resultId: null };

    // Check if LLM is configured, fall back to retrieval-demo then demo mode
    let endpoint = '/query';
    try {
        const statusResp = await fetch('/api/status');
        const statusData = await statusResp.json();
        if (!statusData.llm_configured) {
            endpoint = '/query/retrieval-demo';
        }
    } catch (e) { /* use default */ }

    // Guard against double-submit
    if (submitBtn.disabled) return;

    try {
        const body = (endpoint === '/query/demo' || endpoint === '/query/retrieval-demo')
            ? { question }
            : { question, max_rounds: 5 };
        const MAX_RETRIES = 3;
        const TIMEOUT_MS = parseInt(localStorage.getItem('verarag-timeout') || '120000');
        let attempt = 0;
        let response;

        while (attempt < MAX_RETRIES) {
            attempt++;
            try {
                const ctrl = new AbortController();
                const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
                response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                    signal: ctrl.signal
                });
                clearTimeout(timer);
                if (!response.ok) throw new Error(`服务端错误: ${response.status}`);
                break;
            } catch (err) {
                if (attempt < MAX_RETRIES && (err.name === 'AbortError' || err.message.includes('Failed to fetch'))) {
                    setStageDetail('retrieval', `连接中断，第 ${attempt} 次重试...`);
                    await new Promise(r => setTimeout(r, 2000 * attempt));
                    continue;
                }
                throw err;
            }
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let currentEvent = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line === '') {
                    currentEvent = '';
                    continue;
                }
                if (line.startsWith('event: ')) {
                    currentEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ') && currentEvent) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        handleEvent(currentEvent, data);
                    } catch (err) {
                        // skip malformed JSON
                    }
                    currentEvent = '';
                }
            }
        }
    } catch (err) {
        if (errorPanel) {
            errorPanel.classList.remove('hidden');
            if (err.name === 'AbortError') {
                errorText.textContent = '请求超时，请重试';
            } else if (!navigator.onLine) {
                errorText.textContent = '网络连接已断开，请恢复网络后重试';
            } else {
                errorText.textContent = '连接失败: ' + err.message;
            }
        }
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = '开始推理';
    }
});

function handleEvent(eventType, data) {
    switch (eventType) {
        case 'stage':
            if (data.status === 'started') {
                setStageActive(data.stage);
                if (data.stage === 'retrieval') {
                    setStageDetail('retrieval', `第 ${data.round}/${data.total_rounds} 轮`);
                }
            }
            break;

        case 'task_analysis':
            setStageDone('task_analysis');
            setStageDetail('task_analysis', `${data.task_type} · ${data.complexity}`);
            break;

        case 'decomposition':
            setStageDone('decomposition');
            setStageDetail('decomposition', `${data.subquestions.length} 个子问题`);
            break;

        case 'evidence':
            currentEventData.evidence = data.evidence || currentEventData.evidence;
            setStageDetail('retrieval', `已获取 ${data.total} 条证据`);
            break;

        case 'conflict':
            setStageDetail('retrieval', `${data.conflicts} 个冲突 · 分数 ${data.conflict_score.toFixed(2)}`);
            break;

        case 'uncertainty':
            currentEventData.uncertainty = data;
            break;

        case 'reasoning':
            setStageDone('retrieval');
            setStageDone('reasoning');
            currentEventData.reasoning = data;
            if (data.claims) {
                currentEventData.claims = data.claims;
            }
            // Show answer
            document.getElementById('answer-text').textContent = data.answer || '';
            break;

        case 'verification':
            setStageDone('verification');
            currentEventData.verification = data;
            break;

        case 'repair':
            setStageActive('repair');
            currentEventData.repair = data;
            if (data.repaired_answer) {
                document.getElementById('answer-text').textContent = data.repaired_answer;
            }
            break;

        case 'complete':
            // Finalize
            if (currentEventData.claims && currentEventData.claims.length > 0) {
                setStageDone('repair');
            }
            document.getElementById('confidence-bar').style.width = `${(data.confidence * 100).toFixed(0)}%`;
            document.getElementById('confidence-label').textContent = `置信度 ${(data.confidence * 100).toFixed(0)}%`;

            // Render tabs
            renderEvidence(currentEventData.evidence);
            renderClaims(currentEventData.claims);
            renderConflictGraph(currentEventData);
            renderUncertainty(currentEventData.uncertainty);
            renderRepair(currentEventData.repair);

            resultPanel.classList.remove('hidden');
            break;

        case 'saved':
            currentEventData.resultId = data.result_id;
            break;

        case 'error':
            errorPanel.classList.remove('hidden');
            errorText.textContent = data.error || '未知错误';
            break;

        case 'ping':
            break;
    }
}

// Auto-resize textarea
if (isIndexPage) {
    questionInput.addEventListener('input', () => {
        questionInput.style.height = 'auto';
        questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
    });
}

// File upload
if (isIndexPage) {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadProgress = document.getElementById('upload-progress');
    const uploadBar = document.getElementById('upload-bar');
    const uploadStatus = document.getElementById('upload-status');

    if (uploadArea) {
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); });
        uploadArea.addEventListener('dragleave', () => {});
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => { if (fileInput.files.length) handleUpload(fileInput.files[0]); });
    }

    function handleUpload(file) {
        const suffix = '.' + file.name.split('.').pop().toLowerCase();
        if (!['.pdf', '.txt', '.md'].includes(suffix)) {
            alert('仅支持 PDF、TXT、MD 文件');
            return;
        }
        uploadProgress.classList.remove('hidden');
        uploadBar.style.width = '0%';
        uploadStatus.textContent = `上传中: ${file.name}`;

        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload');
        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) uploadBar.style.width = (e.loaded / e.total * 100) + '%';
        };
        xhr.onload = () => {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                uploadStatus.textContent = `已导入: ${data.chunks} 个片段，${data.chars} 字符`;
                uploadBar.style.width = '100%';
            } else {
                uploadStatus.textContent = '上传失败: ' + xhr.statusText;
            }
        };
        xhr.onerror = () => { uploadStatus.textContent = '上传失败'; };
        xhr.send(formData);
    }
}
