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
    resultId: null
};

// Network status monitoring
const networkBanner = document.createElement('div');
networkBanner.id = 'network-banner';
networkBanner.className = 'fixed top-0 left-0 right-0 bg-red-500/90 text-white text-center text-sm py-2 z-50 hidden';
networkBanner.textContent = '网络连接已断开，请检查网络';
document.body.appendChild(networkBanner);

window.addEventListener('offline', () => {
    networkBanner.classList.remove('hidden');
});
window.addEventListener('online', () => {
    networkBanner.classList.add('hidden');
});

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
    list.innerHTML = claims.map(c => {
        const statusColors = {
            'supported': 'text-emerald-400 bg-emerald-400/10',
            'refuted': 'text-red-400 bg-red-400/10',
            'not_enough_info': 'text-amber-400 bg-amber-400/10'
        };
        const statusClass = statusColors[c.verification_status] || statusColors['not_enough_info'];
        const statusLabel = { 'supported': '已验证', 'refuted': '已反驳', 'not_enough_info': '证据不足' };
        return `
            <div class="bg-white/[0.02] rounded-xl p-4 border border-white/5">
                <p class="text-sm text-gray-300 mb-2">${esc(c.claim)}</p>
                <div class="flex items-center gap-2">
                    <span class="text-xs px-2 py-0.5 rounded-full ${statusClass}">${statusLabel[c.verification_status] || c.verification_status}</span>
                    <span class="text-xs text-gray-600">置信度 ${(c.confidence || 0).toFixed(2)}</span>
                </div>
            </div>
        `;
    }).join('');
}

function renderUncertainty(uncertainty) {
    const el = document.getElementById('uncertainty-detail');
    if (!uncertainty) {
        el.innerHTML = '<p class="text-sm text-gray-500">暂无不确定性信息</p>';
        return;
    }
    const dims = [
        { label: '检索不确定性', value: uncertainty.retrieval_uncertainty || 0 },
        { label: '证据冲突', value: uncertainty.evidence_conflict || 0 },
        { label: '推理差距', value: uncertainty.reasoning_gap || 0 },
        { label: '来源可靠性', value: uncertainty.source_reliability || 0 },
        { label: '验证不确定性', value: uncertainty.verification_uncertainty || 0 }
    ];
    el.innerHTML = `
        <div class="space-y-4">
            ${dims.map(d => `
                <div>
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-gray-400">${d.label}</span>
                        <span class="text-gray-500">${(d.value).toFixed(2)}</span>
                    </div>
                    <div class="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
                        <div class="h-full rounded-full ${d.value > 0.5 ? 'bg-amber-500' : 'bg-emerald-500'}" style="width: ${(d.value * 100).toFixed(0)}%"></div>
                    </div>
                </div>
            `).join('')}
            <div class="pt-3 border-t border-white/5">
                <div class="flex justify-between text-sm">
                    <span class="text-gray-300">总体不确定性</span>
                    <span class="font-medium ${(uncertainty.overall_uncertainty || 0) > 0.5 ? 'text-amber-400' : 'text-emerald-400'}">${(uncertainty.overall_uncertainty || 0).toFixed(2)}</span>
                </div>
            </div>
        </div>
    `;
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
    currentEventData = { evidence: [], claims: [], uncertainty: null, reasoning: null, verification: null, resultId: null };

    // Check if LLM is configured, fall back to demo mode
    let endpoint = '/query';
    try {
        const statusResp = await fetch('/api/status');
        const statusData = await statusResp.json();
        if (!statusData.llm_configured) {
            endpoint = '/query/demo';
        }
    } catch (e) { /* use default */ }

    // Guard against double-submit
    if (submitBtn.disabled) return;

    try {
        const body = endpoint === '/query/demo'
            ? { question }
            : { question, max_rounds: 5 };
        const ctrl = new AbortController();
        const TIMEOUT_MS = parseInt(localStorage.getItem('verarag-timeout') || '120000');
        const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: ctrl.signal
        });
        clearTimeout(timer);

        if (!response.ok) {
            throw new Error(`服务端错误: ${response.status}`);
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
        case 'ping':
            break;

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
            renderUncertainty(currentEventData.uncertainty);

            resultPanel.classList.remove('hidden');
            break;

        case 'saved':
            currentEventData.resultId = data.result_id;
            break;

        case 'error':
            errorPanel.classList.remove('hidden');
            errorText.textContent = data.error || '未知错误';
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
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const uploadProgress = document.getElementById('upload-progress');
const uploadBar = document.getElementById('upload-bar');
const uploadStatus = document.getElementById('upload-status');

if (uploadArea) {
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('border-violet-500/30'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('border-violet-500/30'));
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('border-violet-500/30');
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
