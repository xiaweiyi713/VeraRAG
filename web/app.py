"""FastAPI application factory for VeraRAG Web UI."""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .db import Database
from .api import create_router


def create_app(config_path: Optional[str] = None, db_path: Optional[str] = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config_path: Path to VeraRAG YAML config file
        db_path: Path to SQLite database file

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="VeraRAG",
        description="Verifiable Agentic Retrieval-Augmented Reasoning",
        version="0.1.0"
    )

    # Paths
    web_dir = Path(__file__).parent
    templates_dir = web_dir / "templates"
    static_dir = web_dir / "static"

    # Database
    if db_path is None:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        db_path = str(data_dir / "verarag.db")
    db = Database(db_path)

    # Templates
    templates = Jinja2Templates(directory=str(templates_dir))

    # Static files
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Config
    config = None
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Store shared state
    app.state.db = db
    app.state.templates = templates
    app.state.config = config

    # Routes
    router = create_router(templates, db, config)
    app.include_router(router)

    return app
