let sessionId = null;

async function getJSON(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

function val(id) { return document.getElementById(id).value.trim(); }
function setText(id, text) { document.getElementById(id).textContent = text; }
function setHtml(id, html) { document.getElementById(id).innerHTML = html; }

async function loadOptions() {
  try {
    const opts = await getJSON('/api/meta/options');
    const topicSel = document.getElementById('topic');
    const qtypeSel = document.getElementById('qtype');
    const diffSel = document.getElementById('difficulty');
    for (const t of (opts.topics || [])) topicSel.insertAdjacentHTML('beforeend', `<option>${t}</option>`);
    for (const t of (opts.qtypes || [])) qtypeSel.insertAdjacentHTML('beforeend', `<option>${t}</option>`);
    for (const t of (opts.difficulties || [])) diffSel.insertAdjacentHTML('beforeend', `<option>${t}</option>`);
  } catch (e) {
    console.error(e);
  }
}

async function startSession() {
  const role = val('role') || 'Data Scientist';
  const filters = {};
  const topic = val('topic');
  const qtype = val('qtype');
  const difficulty = val('difficulty');
  if (topic) filters.topic = topic;
  if (qtype) filters.qtype = qtype;
  if (difficulty) filters.difficulty = difficulty;

  const mode = val('mode');
  const total = parseInt(val('total') || '0', 10);
  if (mode === 'mock') {
    const params = new URLSearchParams({ role, total_questions: String(total || 5) });
    if (filters.topic) params.set('topic', filters.topic);
    if (filters.qtype) params.set('qtype', filters.qtype);
    if (filters.difficulty) params.set('difficulty', filters.difficulty);
    const res = await getJSON(`/api/session/start_mock?${params}` , { method: 'POST' });
    sessionId = res.session_id;
  } else {
    const body = { profile: { role }, config: { filters } };
    const res = await getJSON('/api/session/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    sessionId = res.session_id;
  }
  setText('sessionInfo', `Session: ${sessionId}`);
  setText('questionText', '');
  setHtml('questionMeta', '');
  setText('feedback', '');
  setText('progress', '');
  setText('summary', '');
  setText('remaining', '');
}

async function getQuestion(forceNew = false) {
  if (!sessionId) return alert('请先启动会话');
  const url = `/api/session/${sessionId}/question${forceNew ? '?force_new=true' : ''}`;
  const q = await getJSON(url);
  setText('questionText', q.question_text || '');
  setHtml('questionMeta', `ID: <b>${q.question_id}</b> | Topic: <b>${q.topic || '-'}</b> | Type: <b>${q.qtype || '-'}</b> | Difficulty: <b>${q.difficulty || '-'}</b>`);
}

async function submitAnswer() {
  if (!sessionId) return alert('请先启动会话');
  const answer = val('answer');
  if (!answer) return alert('请先输入答案');
  const res = await getJSON(`/api/session/${sessionId}/answer`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ answer_text: answer }) });
  setText('remaining', res.remaining != null ? `剩余题数：${res.remaining}` : '');
  setText('feedback', JSON.stringify(res.feedback, null, 2));
}

async function showProgress() {
  if (!sessionId) return alert('请先启动会话');
  const pr = await getJSON(`/api/session/${sessionId}/progress`);
  setText('progress', JSON.stringify(pr, null, 2));
}

async function showSummary() {
  if (!sessionId) return alert('请先启动会话');
  const sm = await getJSON(`/api/session/${sessionId}/summary`);
  setText('summary', JSON.stringify(sm, null, 2));
}

async function exportHistory() {
  if (!sessionId) return alert('请先启动会话');
  const data = await getJSON(`/api/session/${sessionId}/export`);
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `session_${sessionId}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

document.getElementById('startBtn').addEventListener('click', startSession);
document.getElementById('nextBtn').addEventListener('click', () => getQuestion(false));
document.getElementById('forceBtn').addEventListener('click', () => getQuestion(true));
document.getElementById('submitBtn').addEventListener('click', submitAnswer);
document.getElementById('progressBtn').addEventListener('click', showProgress);
document.getElementById('summaryBtn').addEventListener('click', showSummary);
document.getElementById('exportBtn').addEventListener('click', exportHistory);

loadOptions();

