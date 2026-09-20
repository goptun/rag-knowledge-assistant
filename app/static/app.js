const EXAMPLES = [
  "Como funciona a autenticação da API?",
  "Quais campos são obrigatórios para criar uma cobrança?",
  "Qual o rate limit da API?",
  "Como consultar o status de uma cobrança pelo id?",
];

const questionEl = document.getElementById('question');
const modeEl = document.getElementById('mode');
const askBtn = document.getElementById('ask-btn');
const statusEl = document.getElementById('status');
const resultPanel = document.getElementById('result-panel');
const retrievedEl = document.getElementById('retrieved');
const answerEl = document.getElementById('answer');
const citationsEl = document.getElementById('citations');

// Texto cru acumulado do stream; a cada delta re-renderiza como Markdown.
let answerText = '';

function renderAnswer() {
  answerEl.innerHTML = renderMarkdown(answerText);
}

function showError(message) {
  const p = document.createElement('p');
  p.className = 'error-text';
  p.textContent = message;
  answerEl.appendChild(p);
}

const examplesEl = document.getElementById('examples');
EXAMPLES.forEach((q) => {
  const chip = document.createElement('span');
  chip.className = 'chip';
  chip.textContent = q;
  chip.onclick = () => { questionEl.value = q; };
  examplesEl.appendChild(chip);
});

function setLoading(loading) {
  askBtn.disabled = loading;
  askBtn.textContent = loading ? 'Gerando…' : 'Perguntar';
}

function renderRetrieved(chunks) {
  retrievedEl.innerHTML = '';
  (chunks || []).forEach((c) => {
    const span = document.createElement('span');
    span.className = 'src-chip';
    const loc = c.section || (c.page != null ? `p. ${c.page}` : '');
    span.textContent = loc ? `${c.source} · ${loc}` : c.source;
    retrievedEl.appendChild(span);
  });
}

function renderCitations(citations) {
  citationsEl.innerHTML = '';
  (citations || []).forEach((c) => {
    const div = document.createElement('div');
    div.className = 'citation';
    const loc = c.section || (c.page != null ? `p. ${c.page}` : '');
    div.innerHTML = `<b>[${c.index}]</b> ${c.source}${loc ? ' — ' + loc : ''}`;
    citationsEl.appendChild(div);
  });
}

function handleBlock(block) {
  let eventType = null;
  let data = null;
  for (const line of block.split('\n')) {
    if (line.startsWith('event: ')) eventType = line.slice(7);
    else if (line.startsWith('data: ')) data = JSON.parse(line.slice(6));
  }
  if (!eventType || !data) return;
  if (eventType === 'retrieved') {
    renderRetrieved(data.retrieved);
    statusEl.textContent = `Recuperado com ${data.mode}. Gerando resposta…`;
  } else if (eventType === 'delta') {
    answerText += data.text;
    renderAnswer();
  } else if (eventType === 'done') {
    renderCitations(data.citations);
    statusEl.textContent = '';
  } else if (eventType === 'error') {
    showError(`Erro do modelo: ${data.message}`);
    statusEl.textContent = '';
  }
}

async function ask() {
  const question = questionEl.value.trim();
  if (!question) return;

  resultPanel.hidden = false;
  retrievedEl.innerHTML = '';
  answerText = '';
  answerEl.innerHTML = '';
  citationsEl.innerHTML = '';
  statusEl.textContent = 'Buscando contexto relevante…';
  setLoading(true);

  try {
    const resp = await fetch('query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, mode: modeEl.value }),
    });

    if (resp.status === 429) {
      const err = await resp.json().catch(() => ({}));
      showError(err.detail || 'Muitas perguntas em pouco tempo. Espera um minuto.');
      statusEl.textContent = '';
      return;
    }
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail ? JSON.stringify(err.detail) : `Erro HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        handleBlock(block);
      }
    }
  } catch (e) {
    showError(`Erro: ${e.message}`);
    statusEl.textContent = '';
  } finally {
    setLoading(false);
  }
}

askBtn.addEventListener('click', ask);
questionEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask();
});
