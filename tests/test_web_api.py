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

    def test_config_get_default(self):
        """默认配置应为空"""
        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["provider"], "")

    def test_config_update_and_get(self):
        """更新后应能读取配置（API Key 被遮掩）"""
        self.client.put("/api/config", json={
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_key": "sk-12345678abcdefgh",
            "base_url": ""
        })

        response = self.client.get("/api/config")
        data = response.json()
        self.assertEqual(data["provider"], "deepseek")
        self.assertEqual(data["model"], "deepseek-chat")
        self.assertIn("****", data["api_key"])
        self.assertNotEqual(data["api_key"], "sk-12345678abcdefgh")

    def test_config_clear(self):
        """清除配置后应为空"""
        self.client.put("/api/config", json={
            "provider": "deepseek", "model": "deepseek-chat",
            "api_key": "sk-test", "base_url": ""
        })
        self.client.put("/api/config", json={
            "provider": "", "model": "", "api_key": "", "base_url": ""
        })
        response = self.client.get("/api/config")
        data = response.json()
        self.assertEqual(data["provider"], "")

    def test_status_reflects_config(self):
        """状态 API 应反映 LLM 配置状态"""
        # Default: not configured
        response = self.client.get("/api/status")
        self.assertFalse(response.json()["llm_configured"])

        # After config
        self.client.put("/api/config", json={
            "provider": "openai", "model": "gpt-4o",
            "api_key": "sk-test", "base_url": ""
        })
        response = self.client.get("/api/status")
        data = response.json()
        self.assertTrue(data["llm_configured"])
        self.assertEqual(data["llm_provider"], "openai")

    def test_demo_query_sse(self):
        """演示模式应返回 SSE 事件流"""
        response = self.client.post("/query/demo", json={"question": "What is RAG?"})
        self.assertEqual(response.status_code, 200)
        text = response.text
        self.assertIn("event: task_analysis", text)
        self.assertIn("event: reasoning", text)
        self.assertIn("event: complete", text)
        self.assertIn("event: saved", text)

    def test_demo_saves_to_history(self):
        """演示模式结果应保存到历史"""
        self.client.post("/query/demo", json={"question": "Demo question?"})
        queries = self.app.state.db.list_queries()
        self.assertGreater(len(queries), 0)
        self.assertIn("Demo question", queries[0]["question"])


if __name__ == "__main__":
    unittest.main()
