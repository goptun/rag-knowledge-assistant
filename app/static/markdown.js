// Renderizador de Markdown mínimo pra resposta do LLM. Sem dependências
// (o frontend é estático, sem build) e seguro por construção: todo texto
// é escapado antes de virar HTML, e só as tags geradas aqui chegam ao DOM.
// Suporta o que o modelo costuma emitir: parágrafos, títulos, listas,
// tabelas, blocos de código, `código inline`, **negrito**, *itálico*,
// links http(s) e marcadores de citação [n].

function escapeHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Alguns modelos emitem a citação como 【1】 (colchetes de largura total);
// normaliza pra [1] antes de renderizar.
function normalizeCitations(text) {
  return text.replace(/【(\d+)[^】]*】/g, '[$1]');
}

function renderInline(text) {
  const codes = [];
  let out = text.replace(/`([^`\n]+)`/g, (_, code) => {
    codes.push(code);
    return `\u0000${codes.length - 1}\u0000`;
  });
  out = escapeHtml(out);
  out = out
    .replace(
      /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    )
    .replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\s][^*\n]*?)\*(?!\*)/g, '$1<em>$2</em>')
    .replace(/\[(\d+)\]/g, '<sup class="cite">[$1]</sup>');
  return out.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${escapeHtml(codes[i])}</code>`);
}

const FENCE_RE = /^\s*```\s*([\w+-]*)\s*$/;
const HEADING_RE = /^(#{1,6})\s+(.+?)\s*#*\s*$/;
const UL_RE = /^\s*[-*+]\s+(.*)$/;
const OL_RE = /^\s*(\d+)[.)]\s+(.*)$/;
const TABLE_SEP_RE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

function splitRow(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
}

function isBlockStart(line, next) {
  return (
    FENCE_RE.test(line) ||
    HEADING_RE.test(line) ||
    UL_RE.test(line) ||
    OL_RE.test(line) ||
    (line.includes('|') && next !== undefined && TABLE_SEP_RE.test(next))
  );
}

function renderMarkdown(src) {
  const lines = normalizeCitations(src).split('\n');
  const html = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) { i++; continue; }

    const fence = line.match(FENCE_RE);
    if (fence) {
      const body = [];
      i++;
      // Durante o streaming o bloco pode estar aberto: vai até o fim do texto.
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) body.push(lines[i++]);
      i++;
      const lang = fence[1] ? ` data-lang="${escapeHtml(fence[1])}"` : '';
      html.push(`<pre${lang}><code>${escapeHtml(body.join('\n'))}</code></pre>`);
      continue;
    }

    const heading = line.match(HEADING_RE);
    if (heading) {
      const level = Math.min(heading[1].length + 2, 6);
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      i++;
      continue;
    }

    if (line.includes('|') && i + 1 < lines.length && TABLE_SEP_RE.test(lines[i + 1])) {
      const head = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) rows.push(splitRow(lines[i++]));
      const th = head.map((c) => `<th>${renderInline(c)}</th>`).join('');
      const tr = rows
        .map((r) => `<tr>${r.map((c) => `<td>${renderInline(c)}</td>`).join('')}</tr>`)
        .join('');
      html.push(`<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`);
      continue;
    }

    const ul = line.match(UL_RE);
    const ol = line.match(OL_RE);
    if (ul || ol) {
      const ordered = !!ol;
      const re = ordered ? OL_RE : UL_RE;
      const start = ordered ? Number(ol[1]) : 1;
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(re);
        if (!m) break;
        items.push(renderInline(m[m.length - 1]));
        i++;
        // Linhas de continuação (indentadas, sem marcador) pertencem ao item.
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !UL_RE.test(lines[i]) && !OL_RE.test(lines[i])) {
          items[items.length - 1] += '<br>' + renderInline(lines[i].trim());
          i++;
        }
        // Uma linha em branco entre itens da mesma lista não quebra a lista.
        if (i + 1 < lines.length && !lines[i].trim() && re.test(lines[i + 1])) i++;
      }
      const tag = ordered ? 'ol' : 'ul';
      const startAttr = ordered && start !== 1 ? ` start="${start}"` : '';
      html.push(`<${tag}${startAttr}>${items.map((it) => `<li>${it}</li>`).join('')}</${tag}>`);
      continue;
    }

    const para = [];
    while (i < lines.length && lines[i].trim() && !(para.length && isBlockStart(lines[i], lines[i + 1]))) {
      para.push(renderInline(lines[i].trim()));
      i++;
    }
    html.push(`<p>${para.join('<br>')}</p>`);
  }

  return html.join('');
}
