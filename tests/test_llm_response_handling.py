"""Regression tests for Qwen/SGLang replies; no model server or simulator needed."""

import json
import unittest
from unittest.mock import MagicMock, patch

from capx.llm.client import (
    ModelQueryArgs,
    _parse_content_and_reasoning,
    query_model,
    query_model_streaming,
)
from capx.utils.launch_utils import _extract_code, _parse_multi_turn_decision


class ModelResponseTests(unittest.TestCase):
    def setUp(self):
        self.args = ModelQueryArgs(
            model="Qwen3.8-Flash-Next",
            server_url="http://localhost:30000/v1/chat/completions",
            api_key="EMPTY",
            temperature=0.5,
            max_tokens=20480,
        )
        self.prompt = [{"role": "user", "content": "Write Python code."}]

    @staticmethod
    def response(content, **message_fields):
        response = MagicMock()
        response.status_code = 200
        response.ok = True
        response.headers = {"content-type": "application/json"}
        response.__enter__.return_value = response
        response.json.return_value = {
            "choices": [{
                "message": {"content": content, **message_fields},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 100},
        }
        return response

    def payload(self, post):
        try:
            return json.loads(post.call_args.kwargs["data"])
        except (TypeError, json.JSONDecodeError) as exc:
            self.fail(f"Client sent invalid JSON: {exc}")

    def test_sglang_reasoning_is_saved_separately(self):
        response = self.response("print('ok')", reasoning_content="Plan the action.")
        with patch("capx.llm.client.requests.post", return_value=response):
            result = query_model(self.args, self.prompt)
        self.assertEqual(result["content"], "print('ok')")
        self.assertEqual(result["reasoning"], "Plan the action.")

    def test_reasoning_only_reply_fails_with_token_diagnostics(self):
        for content in ("", None, " \n "):
            with self.subTest(content=content):
                response = self.response(content, reasoning_content="Still thinking...")
                body = response.json.return_value
                body["choices"][0]["finish_reason"] = "length"
                body["usage"] = {"completion_tokens": 20480, "reasoning_tokens": 20480}
                with patch("capx.llm.client.requests.post", return_value=response):
                    with self.assertRaisesRegex(RuntimeError, "empty final content.*length.*20480"):
                        query_model(self.args, self.prompt)

    def test_thinking_override_does_not_change_other_settings(self):
        for enable_thinking in (None, False, True):
            with self.subTest(enable_thinking=enable_thinking):
                self.args.enable_thinking = enable_thinking
                with patch("capx.llm.client.requests.post", return_value=self.response("pass")) as post:
                    query_model(self.args, self.prompt)
                payload = self.payload(post)
                self.assertEqual(payload["temperature"], 0.5)
                self.assertEqual(payload["max_tokens"], 20480)
                self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer EMPTY")
                if enable_thinking is None:
                    self.assertNotIn("chat_template_kwargs", payload)
                else:
                    self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": enable_thinking})

    def test_qwen_thinking_effort_defaults_and_overrides(self):
        self.assertEqual(self.args.thinking_effort, "low")
        for query in (query_model, query_model_streaming):
            for effort in ("low", "medium", "xhigh"):
                for enabled in (None, True, False):
                    with self.subTest(query=query.__name__, effort=effort, enabled=enabled):
                        self.args.thinking_effort = effort
                        self.args.enable_thinking = enabled
                        with patch("capx.llm.client.requests.post", return_value=self.response("pass")) as post:
                            result = query(self.args, self.prompt)
                            if query is query_model_streaming:
                                list(result)
                        payload = self.payload(post)
                        self.assertEqual(payload["reasoning_effort"], effort)
                        self.assertEqual(payload["max_tokens"], 20480)
                        if enabled is None:
                            self.assertNotIn("chat_template_kwargs", payload)
                        else:
                            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": enabled})

    def test_qualified_qwen_model_name_uses_low_effort(self):
        self.args.model = "Qwen/Qwen3.8-Flash-Next"
        with patch("capx.llm.client.requests.post", return_value=self.response("pass")) as post:
            query_model(self.args, self.prompt)
        self.assertEqual(self.payload(post)["reasoning_effort"], "low")

    def test_other_models_keep_existing_reasoning_settings(self):
        for query in (query_model, query_model_streaming):
            for model in ("qwen3.5-122b-a10b-w8a8", "openai/gpt-5.4"):
                with self.subTest(query=query.__name__, model=model):
                    self.args.model = model
                    self.args.thinking_effort = "xhigh"
                    with patch("capx.llm.client.requests.post", return_value=self.response("pass")) as post:
                        result = query(self.args, self.prompt)
                        if query is query_model_streaming:
                            list(result)
                    payload = self.payload(post)
                    if model == "openai/gpt-5.4":
                        self.assertEqual(payload["reasoning_effort"], "medium")
                    else:
                        self.assertNotIn("reasoning_effort", payload)

    def test_invalid_qwen_effort_fails_before_network_request(self):
        self.args.__dict__["thinking_effort"] = "high"
        with patch("capx.llm.client.requests.post") as post:
            with self.assertRaisesRegex(ValueError, "low, medium, or xhigh"):
                query_model(self.args, self.prompt)
            post.assert_not_called()

    def test_launch_defaults_and_effort_values(self):
        from capx.envs.launch import LaunchArgs
        from capx.envs.scripts.run_libero_batch import LiberoBatchLaunchArgs

        self.assertEqual(LaunchArgs(config_path="unused.yaml").thinking_effort, "low")
        self.assertEqual(LiberoBatchLaunchArgs().thinking_effort, "low")
        self.assertEqual(LiberoBatchLaunchArgs().max_tokens, 32768)
        for effort in ("low", "medium", "xhigh"):
            self.assertEqual(LiberoBatchLaunchArgs(thinking_effort=effort).thinking_effort, effort)
            self.assertEqual(LaunchArgs(config_path="unused.yaml", thinking_effort=effort).thinking_effort, effort)

    def test_legacy_reasoning_and_embedded_think_tags(self):
        self.assertEqual(_parse_content_and_reasoning(None), ("", None))
        self.assertEqual(
            _parse_content_and_reasoning("<think>Plan</think>print(1)"),
            ("print(1)", "Plan"),
        )
        response = self.response("pass", reasoning="Legacy reasoning")
        with patch("capx.llm.client.requests.post", return_value=response):
            self.assertEqual(query_model(self.args, self.prompt)["reasoning"], "Legacy reasoning")

    def test_streaming_json_fallback_and_empty_reply(self):
        response = self.response("pass", reasoning_content="Plan")
        with patch("capx.llm.client.requests.post", return_value=response):
            events = list(query_model_streaming(self.args, self.prompt))
        self.assertEqual(events[-1], {"type": "done", "content": "pass", "reasoning": "Plan"})
        with patch("capx.llm.client.requests.post", return_value=self.response(None, reasoning_content="Plan")):
            with self.assertRaisesRegex(RuntimeError, "empty final content"):
                list(query_model_streaming(self.args, self.prompt))

    def test_streaming_sse_reasoning_content(self):
        response = self.response("unused")
        response.headers = {"content-type": "text/event-stream"}
        response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"reasoning_content":"Plan"}}]}',
            b'data: {"choices":[{"delta":{"content":"pass"}}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            b'data: [DONE]',
        ]
        self.args.enable_thinking = False
        with patch("capx.llm.client.requests.post", return_value=response) as post:
            events = list(query_model_streaming(self.args, self.prompt))
        self.assertEqual(events[0], {"type": "reasoning_delta", "content": "Plan"})
        self.assertEqual(events[-1], {"type": "done", "content": "pass", "reasoning": "Plan"})
        self.assertEqual(self.payload(post)["chat_template_kwargs"], {"enable_thinking": False})

    def test_reasoning_only_sse_does_not_emit_done(self):
        response = self.response("unused")
        response.headers = {"content-type": "text/event-stream"}
        response.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"reasoning_content":"Plan"}}]}',
            b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}',
            b'data: {"choices":[],"usage":{"completion_tokens":20480}}',
            b'data: [DONE]',
        ]
        events = []
        with patch("capx.llm.client.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "empty final content.*length.*20480"):
                for event in query_model_streaming(self.args, self.prompt):
                    events.append(event)
        self.assertFalse(any(event["type"] == "done" for event in events))

    def test_empty_code_and_decisions_are_not_successful_noops(self):
        for content in ("", " \n", "```python\n\n```"):
            with self.subTest(code=content), self.assertRaisesRegex(ValueError, "empty Python"):
                _extract_code(content)
        for content in (None, "", " \n"):
            with self.subTest(decision=content), self.assertRaisesRegex(ValueError, "Empty multi-turn"):
                _parse_multi_turn_decision(content)
        self.assertEqual(_extract_code("```python\npass\n```"), ["pass"])
        self.assertEqual(_parse_multi_turn_decision("FINISH"), ("finish", "FINISH"))
        self.assertEqual(_parse_multi_turn_decision("REGENERATE\npass"), ("regenerate", "pass"))


if __name__ == "__main__":
    unittest.main()
