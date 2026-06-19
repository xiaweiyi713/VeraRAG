#!/usr/bin/env python3
"""VeraRAG Web UI launcher."""

import argparse
import os
import sys

import uvicorn

sys.path.insert(0, os.path.dirname(__file__))


def main():
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
    else:
        from web.app import create_app

        app = create_app(config_path=args.config, db_path=args.db)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
