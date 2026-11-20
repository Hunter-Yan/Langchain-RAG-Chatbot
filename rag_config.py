import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "books"
CHROMA_PATH = PROJECT_ROOT / "chroma"
EMBEDDING_MODEL = "text-embedding-3-small"


def get_embeddings():
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("Set OPENAI_API_KEY in the project's .env file first.")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)
