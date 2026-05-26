"""Tests for web API endpoints."""

import os
import sys
import unittest
import tempfile

sys.path.insert(0, 'src')

from fastapi.testclient import TestClient
from web.app import create_app


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoints."""

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.tmpfile.close()
        self.app = create_app(db_path=self.tmpfile.name)
        self.client = TestClient(self.app)

    def tearDown(self):
        os.unlink(self.tmpfile.name)

    def test_home_page(self):
        """首页应返回 200"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("VeraRAG", response.text)

    def test_history_page(self):
        """历史页应返回 200"""
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)

    def test_detail_page_not_found(self):
        """不存在的查询详情应返回 404"""
        response = self.client.get("/history/nonexistent123")
        self.assertEqual(response.status_code, 404)

    def test_api_status(self):
        """系统状态 API 应返回 JSON"""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)

    def test_detail_page_exists(self):
        """存在的查询详情应返回 200"""
        db = self.app.state.db
        query_id = db.save_query("Test Q?", "Test A.", 0.75, {"key": "value"})

        response = self.client.get(f"/history/{query_id}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Test Q?", response.text)


if __name__ == "__main__":
    unittest.main()
