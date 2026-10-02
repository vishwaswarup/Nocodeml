"""ASGI entrypoint:  uvicorn nocodeml_engine.api.main:app --reload"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # repo-root .env, git-ignored

from nocodeml_engine.api.app import create_app  # noqa: E402

app = create_app()
