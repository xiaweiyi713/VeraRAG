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


def main() -> None:
    """Run the VeraRAG Web UI from the installed console script."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="VeraRAG Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--config", default="configs/model.yaml", help="Path to config YAML file")
    parser.add_argument("--db", default=None, help="Path to SQLite database file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    print("\n  VeraRAG Web UI")
    print(f"  http://{args.host}:{args.port}\n")

    if args.reload:
        uvicorn.run(
            "web.app:create_app",
            host=args.host,
            port=args.port,
            reload=True,
            factory=True,
            log_level="info",
        )
        return

    app = create_app(config_path=args.config, db_path=args.db)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
