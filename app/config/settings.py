"""Configuração central do projeto via variáveis de ambiente."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_base"

    # Embeddings / Reranker
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Índice sparse (BM25), caminho relativo à raiz do projeto
    bm25_index_path: str = "data/bm25_index.pkl"

    # LLM
    # "anthropic" (API oficial, precisa de crédito) ou "openai_compatible"
    # (proxy local tipo 9Router/LiteLLM/Ollama, via LLM_BASE_URL)
    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Geração (Fase 5)
    llm_model: str = "claude-sonnet-4-5-20250929"
    llm_base_url: str = ""
    llm_api_key: str = ""
    generation_max_tokens: int = 1024
    generation_temperature: float = 0.0


settings = Settings()
