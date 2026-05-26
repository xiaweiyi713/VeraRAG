"""SQLite database layer for query history."""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


class Database:
    """SQLite database for storing query history and config."""

    def __init__(self, db_path: str = "data/verarag.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def save_query(
        self,
        question: str,
        answer: str,
        confidence: float,
        result_json: Any
    ) -> str:
        """Save a query result and return its ID."""
        query_id = uuid.uuid4().hex[:12]
        json_str = json.dumps(result_json, ensure_ascii=False)

        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO queries (id, question, answer, confidence, result_json) VALUES (?, ?, ?, ?, ?)",
                (query_id, question, answer, confidence, json_str)
            )
            conn.commit()
        return query_id

    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get a single query by ID."""
        conn = self._get_conn()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM queries WHERE id = ?", (query_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "confidence": row["confidence"],
                "result_json": json.loads(row["result_json"]),
                "created_at": row["created_at"]
            }
        finally:
            conn.close()

    def list_queries(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List queries ordered by most recent first."""
        conn = self._get_conn()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM queries ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row["answer"][:100] + "..." if len(row["answer"]) > 100 else row["answer"],
                    "confidence": row["confidence"],
                    "created_at": row["created_at"]
                }
                for row in rows
            ]
        finally:
            conn.close()

    def delete_query(self, query_id: str) -> bool:
        """Delete a query by ID."""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM queries WHERE id = ?", (query_id,))
            conn.commit()
            return cursor.rowcount > 0

    # --- Config operations ---

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a config value by key."""
        conn = self._get_conn()
        try:
            cursor = conn.execute("SELECT value FROM config WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row is None:
                return default
            return json.loads(row[0])
        finally:
            conn.close()

    def set_config(self, key: str, value: Any) -> None:
        """Set a config value."""
        json_str = json.dumps(value, ensure_ascii=False)
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                (key, json_str)
            )
            conn.commit()

    def get_llm_config(self) -> Dict[str, Any]:
        """Get the full LLM configuration."""
        return self.get_config("llm_config", {
            "provider": "",
            "model": "",
            "api_key": "",
            "base_url": ""
        })

    def set_llm_config(self, provider: str, model: str, api_key: str, base_url: str = "") -> None:
        """Save LLM configuration."""
        self.set_config("llm_config", {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "base_url": base_url
        })
