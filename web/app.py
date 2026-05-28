"""FastAPI application factory for VeraRAG Web UI."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .api import create_router
from .db import Database

logger = logging.getLogger("verarag")


def create_app(config_path: str | None = None, db_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="VeraRAG",
        description="Verifiable Agentic Retrieval-Augmented Reasoning",
        version="0.1.0",
        docs_url=None,
        redoc_url=None
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

    # Config
    config = None
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Store shared state
    app.state.db = db
    app.state.templates = templates
    app.state.config = config

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}", exc_info=True)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "内部错误"}, status_code=500)
        return JSONResponse({"error": "内部错误"}, status_code=500)

    # Favicon - redirect to prevent 404 noise
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return RedirectResponse(url="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>V</text></svg>")

    # Routes
    router = create_router(templates, db, config)
    app.include_router(router)

    # Static files (mount last so routes take priority)
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app
