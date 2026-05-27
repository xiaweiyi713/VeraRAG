"""Tests for web database layer."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, 'src')

from web.db import Database


class TestDatabase(unittest.TestCase):
    """Test SQLite database operations."""

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix='.db', delete=False)  # noqa: SIM115
        self.tmpfile.close()
        self.db = Database(self.tmpfile.name)

    def tearDown(self):
        os.unlink(self.tmpfile.name)

    def test_init_creates_table(self):
        """初始化时应自动创建 queries 表"""
        import sqlite3
        conn = sqlite3.connect(self.tmpfile.name)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='queries'"
        )
        self.assertIsNotNone(cursor.fetchone())
        conn.close()

    def test_save_and_get_query(self):
        """保存查询后应能通过 ID 取回"""
        result = {
            "question": "What is RAG?",
            "answer": "RAG is a technique...",
            "confidence": 0.85,
            "full_result": {"answer": "RAG is a technique...", "confidence": 0.85}
        }
        query_id = self.db.save_query(
            question=result["question"],
            answer=result["answer"],
            confidence=result["confidence"],
            result_json=result["full_result"]
        )
        self.assertIsNotNone(query_id)

        retrieved = self.db.get_query(query_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["question"], "What is RAG?")
        self.assertEqual(retrieved["answer"], "RAG is a technique...")
        self.assertAlmostEqual(retrieved["confidence"], 0.85)

    def test_list_queries(self):
        """应能列出所有查询"""
        self.db.save_query("Q1", "A1", 0.8, {})
        self.db.save_query("Q2", "A2", 0.9, {})

        queries = self.db.list_queries()
        self.assertEqual(len(queries), 2)
        # 按时间倒序
        self.assertEqual(queries[0]["question"], "Q2")

    def test_list_queries_with_limit(self):
        """应支持分页限制"""
        for i in range(5):
            self.db.save_query(f"Q{i}", f"A{i}", 0.5, {})
        queries = self.db.list_queries(limit=3)
        self.assertEqual(len(queries), 3)

    def test_get_nonexistent_query(self):
        """查询不存在的 ID 应返回 None"""
        result = self.db.get_query("nonexistent")
        self.assertIsNone(result)

    def test_delete_query(self):
        """删除后不应能再取回"""
        query_id = self.db.save_query("Q", "A", 0.5, {})
        self.db.delete_query(query_id)
        result = self.db.get_query(query_id)
        self.assertIsNone(result)

    def test_api_key_encryption(self):
        """API keys should be stored encrypted in SQLite."""
        import sqlite3

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(os.path.join(tmpdir, "test.db"))
            db.set_llm_config("openai", "gpt-4o", "sk-test-secret-key-12345")

            conn = sqlite3.connect(db.db_path)
            row = conn.execute("SELECT value FROM config WHERE key = 'llm_config'").fetchone()
            conn.close()
            raw_value = row[0]
            assert "sk-test-secret-key-12345" not in raw_value

            cfg = db.get_llm_config()
            assert cfg["api_key"] == "sk-test-secret-key-12345"


if __name__ == "__main__":
    unittest.main()
