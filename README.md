# RAG-Based Knowledge Assistant

Assistente de conhecimento interno baseado em RAG (Retrieval-Augmented Generation), com retrieval híbrido (vetorial + keyword) e reranking, e avaliação comparativa entre estratégias de retrieval.

## Status

Fases 1-4 (ingestão, indexação, hybrid retrieval + reranking, avaliação de retrieval) e Fase 5 (API FastAPI + geração via LLM + Answer Faithfulness) implementadas e validadas. Fases seguintes (deploy, polish) estão descritas no plano de implementação, ainda não implementadas.

## Arquitetura

```
Documents → Parsing → Chunking → Embeddings → Vector DB → Retrieval → Reranking → LLM → Cited response
```

## Stack

- **Orquestração**: LlamaIndex
- **Vector DB**: Qdrant (Docker)
- **Embeddings**: BAAI/bge-base-en-v1.5 (local)
- **Reranker**: BAAI/bge-reranker-base (local, cross-encoder)
- **API**: FastAPI (streaming via SSE)
- **LLM de geração**: Claude (Anthropic), citação de fontes por chunk
- **Deploy**: Docker Compose (VPS)

## Estrutura do projeto

```
app/
  parsing/      # extração de texto de PDF/HTML/Markdown (Fase 1)
  chunking/     # recursive e semantic chunking (Fase 1)
  indexing/     # embeddings + upsert no Qdrant (Fase 2)
  retrieval/    # hybrid search (RRF) + reranking (Fase 3)
  evaluation/   # métricas de retrieval (Fase 4) + Answer Faithfulness via RAGAS (Fase 5)
  generation/   # prompt + cliente LLM (streaming) + citações (Fase 5)
  api/          # endpoints FastAPI: /query (streaming) e /health (Fase 5)
  config/       # settings via variáveis de ambiente
data/
  test_docs/    # documentos sintéticos para validar o pipeline
  eval/         # dataset de perguntas + ground truth (Fase 4)
docker/
  docker-compose.yml  # Qdrant + API para desenvolvimento local
  Dockerfile          # imagem da API (Fase 5)
scripts/
  run_ingestion.py       # valida parsing + chunking ponta a ponta (Fase 1)
  run_indexing.py        # parse -> chunk -> embed -> upsert no Qdrant + BM25 (Fase 2)
  run_query.py            # consulta com hybrid retrieval + reranking (Fase 3)
  run_evaluation.py       # compara os 4 modos de retrieval (Fase 4)
  run_api.py              # sobe a API FastAPI (Fase 5)
  run_generation_eval.py  # Answer Faithfulness via RAGAS (Fase 5)
  generate_test_pdf.py   # gera o PDF de teste sintético
tests/          # testes unitários (unittest, sem dependência de pytest)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Sobe o Qdrant para desenvolvimento local:

```bash
docker compose -f docker/docker-compose.yml up -d
```

## Rodando a Fase 1 (parsing + chunking)

```bash
python scripts/run_ingestion.py --dir data/test_docs --strategy both
```

Compara as duas estratégias de chunking com um chunk_size menor, pra ver a diferença de comportamento (recursive corta no meio de seções grandes, semantic respeita os headings):

```bash
python scripts/run_ingestion.py --dir data/test_docs --strategy both --chunk-size 60 --overlap 10
```

## Rodando a Fase 2 (indexação no Qdrant)

Smoke test sem nenhuma dependência pesada instalada (usa embedder e vector store falsos — só valida a orquestração):

```bash
python scripts/run_indexing.py --dir data/test_docs --dry-run
```

Indexação de verdade (precisa do venv com `pip install -r requirements.txt` e do Qdrant no ar via `docker compose -f docker/docker-compose.yml up -d`):

```bash
python scripts/run_indexing.py --dir data/test_docs
```

Na primeira execução, o `sentence-transformers` baixa o modelo `BAAI/bge-base-en-v1.5` (~400MB) do Hugging Face — pode demorar alguns minutos.

## Rodando a Fase 3 (hybrid retrieval + reranking)

Reindexe primeiro para gerar o índice BM25 (o `run_indexing.py` da Fase 2 já constrói e salva em `data/bm25_index.pkl` por padrão):

```bash
python scripts/run_indexing.py --dir data/test_docs
```

Smoke test sem dependências pesadas:

```bash
python scripts/run_query.py --dry-run
```

Consulta de verdade, comparando os modos (a mesma comparação da Fase 4 de avaliação):

```bash
python scripts/run_query.py --query "Qual o rate limit da API?" --mode vector_only
python scripts/run_query.py --query "Qual o rate limit da API?" --mode hybrid
python scripts/run_query.py --query "Qual o rate limit da API?" --mode hybrid_rerank
python scripts/run_query.py --query "Qual o rate limit da API?" --mode hybrid_rerank_blend
```

**`hybrid_rerank_blend`** é uma mitigação descoberta na Fase 4: o cross-encoder puro (`hybrid_rerank`) às vezes derruba uma passagem correta porque julga o *tema geral* do trecho, não o fato específico nele (ver `app/retrieval/blend.py` pro caso concreto que motivou isso). Em vez de deixar o rerank_score sobrescrever 100% o ranking, esse modo combina (normalizado) o rerank_score com o rrf_score original — o consenso do retrieval híbrido funciona como uma rede de segurança. `rerank_blend_alpha` (default 0.7) controla o peso: 1.0 = igual a `hybrid_rerank`, 0.0 = ignora o cross-encoder.

## Rodando a Fase 4 (avaliação de retrieval)

Smoke test sem dependências pesadas:

```bash
python scripts/run_evaluation.py --dry-run
```

Avaliação de verdade (precisa do Qdrant indexado + BM25 salvo — rode `run_indexing.py` primeiro):

```bash
python scripts/run_evaluation.py
```

Compara `vector_only`, `hybrid`, `hybrid_rerank` e `hybrid_rerank_blend` em Precision@5, Recall@5 e MRR sobre as 26 perguntas em `data/eval/qa_dataset.json` (19 com 1 chunk relevante, 7 multi-hop com 2-4 chunks relevantes). É o gráfico central da apresentação do projeto — mostra empiricamente se hybrid retrieval, reranking e a mitigação de blend valem o custo extra.

**Resultado real (k=5, n=26):**

| modo | precision@5 | recall@5 | MRR |
|---|---|---|---|
| vector_only | 0.254 | 0.968 | 0.859 |
| hybrid | 0.254 | 0.968 | 0.897 |
| hybrid_rerank | 0.254 | 0.958 | 0.904 |
| **hybrid_rerank_blend** | **0.262** | **0.978** | **0.904** |

`--verbose` revelou uma regressão de recall pontual em `hybrid_rerank` (uma pergunta multi-hop perdeu um chunk relevante porque o cross-encoder julgou o tema geral do trecho, não o fato específico nele — ver `app/retrieval/blend.py`). `hybrid_rerank_blend` foi implementado como mitigação e corrigiu a regressão sem sacrificar o ganho de MRR do reranking puro, além de melhorar precision@5 e recall@5 agregados. É o modo default do projeto (`app/retrieval/pipeline.py` e `app/api/dependencies.py`).

## Rodando a Fase 5 (API + geração via LLM)

Precisa de `ANTHROPIC_API_KEY` configurada no `.env` (a geração usa Claude via API — ver decisão de stack no plano de implementação: "pra portfolio, API é mais simples e o foco de valor está no retrieval, não no LLM em si").

Sobe a API (precisa do Qdrant indexado + BM25 salvo, como na Fase 3/4):

```bash
python scripts/run_api.py --reload
```

No boot, a API carrega os modelos locais (embedder + reranker) uma única vez — pode demorar alguns segundos.

Testa o endpoint `/query`, que responde via Server-Sent Events (streaming) e citação de fontes no payload final:

```bash
curl -N -X POST http://localhost:8000/query \
    -H 'Content-Type: application/json' \
    -d '{"question": "Como funciona a autenticação da API?"}'
```

Cada afirmação da resposta é citada como `[n]`, resolvido de volta pros metadados reais do chunk (`source`/`section`/`page`) no evento `done` — sem confiar no LLM para relatar a fonte corretamente (ver `app/generation/citations.py`).

`GET /health` reporta se o índice BM25 foi carregado (hybrid retrieval disponível) e qual collection do Qdrant está em uso.

### Answer Faithfulness via RAGAS

Diferente da Fase 4 (avalia só retrieval), esta métrica avalia se a resposta *gerada* é fiel ao contexto recuperado — detecta alucinação. Tem custo real de API por pergunta (1 chamada pra gerar a resposta + 1+ do juiz do RAGAS), por isso `--limit` controla quantas perguntas rodar:

```bash
python scripts/run_generation_eval.py --dry-run          # valida a orquestração, sem custo de API
python scripts/run_generation_eval.py --limit 5           # roda de verdade nas 5 primeiras perguntas
```

O juiz do RAGAS usa o mesmo provider (Anthropic) configurado pra geração, via `LangchainLLMWrapper` em torno de um `ChatAnthropic` — evita depender de uma chave da OpenAI só pra avaliação.

## Testes

```bash
python -m unittest discover -s tests -v
```

## Nota sobre o ambiente de desenvolvimento

O contador de tokens usado no chunking (`app/chunking/token_utils.py`) usa um tokenizador baseado em regex em vez de tiktoken/BPE. Isso porque chunk_size é um parâmetro heurístico — não existe uma contagem "exata" de tokens que sirva para todo modelo de embedding (bge-base usa WordPiece, não o BPE da OpenAI) — e um contador sem dependências externas evita acoplar o chunking a uma biblioteca específica. Se quiser a contagem exata do tokenizer do embedding model, troque por `AutoTokenizer.from_pretrained(EMBEDDING_MODEL)` do `transformers`.

## Próximos passos

Ver o plano de implementação completo (Fases 6-7: deploy via Docker Compose na VPS, README final) na página do projeto no Notion.
