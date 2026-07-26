const ingestBtn = document.getElementById('ingestBtn');
const ingestStatus = document.getElementById('ingestStatus');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const chatHistory = document.getElementById('chatHistory');
const collectionInput = document.getElementById('collectionName');

// The FastAPI backend is running on the same host and port (since it serves these static files)
// For local development or Render, we just hit the origin relative path.
const API_BASE = window.location.origin;

// Ingest Text Logic
ingestBtn.addEventListener('click', async () => {
    const docId = document.getElementById('docId').value.trim();
    const text = document.getElementById('docText').value.trim();
    const collection = collectionInput.value.trim();

    if (!docId || !text) {
        showStatus('Please provide both Document ID and Content.', 'error');
        return;
    }

    ingestBtn.disabled = true;
    ingestBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Ingesting...';
    
    try {
        const response = await fetch(`${API_BASE}/ingest/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ doc_id: docId, text: text })
        });
        
        if (response.ok) {
            showStatus('Document successfully embedded and indexed!', 'success');
            document.getElementById('docText').value = '';
        } else {
            const err = await response.json();
            showStatus(`Error: ${err.detail || 'Ingestion failed'}`, 'error');
        }
    } catch (error) {
        showStatus(`Network Error: ${error.message}`, 'error');
    } finally {
        ingestBtn.disabled = false;
        ingestBtn.innerHTML = '<i class="fa-solid fa-cloud-arrow-up"></i> Ingest Text';
    }
});

function showStatus(msg, type) {
    ingestStatus.textContent = msg;
    ingestStatus.className = `status-msg ${type}`;
    setTimeout(() => { ingestStatus.textContent = ''; }, 4500);
}

// Chat Logic
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // 1. Add User Message to UI
    appendMessage('user', '<i class="fa-solid fa-user"></i>', query);
    chatInput.value = '';
    
    // 2. Add Loading Indicator
    const loaderId = appendMessage('agent', '<i class="fa-solid fa-brain"></i>', '<i class="fa-solid fa-circle-notch fa-spin"></i> Running self-correcting agent loop...');

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query: query,
                collection_name: collectionInput.value.trim() || 'documents',
                top_k: 4
            })
        });

        if (!response.ok) throw new Error('API request failed');
        
        const data = await response.json();
        
        // 3. Format Agent Response
        let htmlContent = '';
        
        // Add Execution Logs
        if (data.agent_loop_logs && data.agent_loop_logs.length > 0) {
            htmlContent += `<div class="log-box">${data.agent_loop_logs.join('\n')}</div>`;
        }
        
        // Add Answer
        // Replace newlines with <br> for HTML rendering
        const answerText = data.answer.replace(/\n/g, '<br>');
        htmlContent += `<div class="answer" style="margin-top: 1rem; font-size: 1.05rem;">${answerText}</div>`;
        
        // Add Sources
        if (data.source_nodes && data.source_nodes.length > 0) {
            htmlContent += `<div style="margin-top: 1.5rem; font-size: 0.85rem; color: var(--text-muted);"><i class="fa-solid fa-layer-group"></i> Retrieved Sources:</div>`;
            data.source_nodes.forEach((node, i) => {
                const scoreDisplay = node.score ? (node.score).toFixed(3) : 'N/A';
                htmlContent += `
                    <div class="source-card">
                        <strong>Source [${i+1}] (Relevance Score: ${scoreDisplay})</strong>
                        ${node.document_text}
                    </div>
                `;
            });
        }
        
        updateMessage(loaderId, htmlContent);

    } catch (error) {
        updateMessage(loaderId, `<span style="color: var(--error);"><i class="fa-solid fa-triangle-exclamation"></i> Error generating response. Ensure API is running.</span>`);
    }
}

function appendMessage(role, avatarIcon, htmlContent) {
    const msgId = 'msg-' + Date.now();
    const wrapper = document.createElement('div');
    wrapper.className = `message ${role}-msg`;
    wrapper.id = msgId;
    
    wrapper.innerHTML = `
        <div class="avatar">${avatarIcon}</div>
        <div class="msg-content">${htmlContent}</div>
    `;
    
    chatHistory.appendChild(wrapper);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return msgId;
}

function updateMessage(msgId, newHtml) {
    const wrapper = document.getElementById(msgId);
    if (wrapper) {
        wrapper.querySelector('.msg-content').innerHTML = newHtml;
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});
