"""Endpoint routing regression tests; no model server or simulator required."""

import json
import unittest
from unittest.mock import MagicMock, patch

from capx.llm.client import ModelQueryArgs, query_model, query_model_streaming


class ModelEndpointRoutingTests(unittest.TestCase):
    endpoint = "http://10.1.110.29:30000/v1/chat/completions"
    prompt = [{"role": "user", "content": "Reply with pass."}]

    @staticmethod
    def response():
        response = MagicMock()
        response.status_code = 200
        response.ok = True
        response.headers = {"content-type": "application/json"}
        response.__enter__.return_value = response
        response.json.return_value = {
            "choices": [{"message": {"content": "pass"}, "finish_reason": "stop"}]
        }
        return response

    def payload(self, post):
        try:
            return json.loads(post.call_args.kwargs["data"])
        except (TypeError, json.JSONDecodeError) as exc:
            self.fail(f"Client sent invalid JSON: {exc}")

    def run_model(self, query, args):
        result = query(args, self.prompt)
        if query is query_model_streaming:
            return list(result)
        return result

    def test_qwen_bypasses_old_endpoints_and_uses_served_model_id(self):
        for query in (query_model, query_model_streaming):
            for model in ("Qwen3.8-Flash-Next", "Qwen/Qwen3.8-Flash-Next"):
                for old_endpoint in (
                    "http://127.0.0.1:8110/chat/completions",
                    "http://183.230.173.202:36789/v1/chat/completions",
                ):
                    with self.subTest(query=query.__name__, model=model, url=old_endpoint):
                        args = ModelQueryArgs(model=model, server_url=old_endpoint)
                        with patch("capx.llm.client.requests.post", return_value=self.response()) as post:
                            self.run_model(query, args)
                        self.assertEqual(post.call_args.args[0], self.endpoint)
                        payload = self.payload(post)
                        self.assertEqual(payload["model"], "Qwen3.8-Flash-Next")
                        self.assertEqual(payload["reasoning_effort"], "low")
                        self.assertEqual(args.server_url, old_endpoint)
                        self.assertEqual(args.model, model)

    def test_context_length_retry_keeps_direct_endpoint(self):
        for query in (query_model, query_model_streaming):
            with self.subTest(query=query.__name__):
                args = ModelQueryArgs(
                    model="Qwen3.8-Flash-Next",
                    server_url="http://127.0.0.1:8110/chat/completions",
                )
                error = MagicMock()
                error.status_code = 400
                error.json.return_value = {
                    "error": {"message": "maximum context length is 4096 tokens; "
                              "prompt contains at least 2048 input tokens"}
                }
                with patch("capx.llm.client.requests.post", side_effect=[error, self.response()]) as post:
                    self.run_model(query, args)
                self.assertEqual(post.call_count, 2)
                for call in post.call_args_list:
                    self.assertEqual(call.args[0], self.endpoint)
                self.assertEqual(self.payload(post)["max_tokens"], 1536)

    def test_other_models_keep_configured_endpoint(self):
        for query in (query_model, query_model_streaming):
            for model in ("qwen3.5-122b-a10b-w8a8", "deepseek-v4-flash-s1", "openai/gpt-5.4"):
                with self.subTest(query=query.__name__, model=model):
                    args = ModelQueryArgs(model=model, server_url="http://other-server/v1/chat/completions")
                    with patch("capx.llm.client.requests.post", return_value=self.response()) as post:
                        self.run_model(query, args)
                    self.assertEqual(post.call_args.args[0], args.server_url)
                    self.assertEqual(self.payload(post)["model"], model)


if __name__ == "__main__":
    unittest.main()
