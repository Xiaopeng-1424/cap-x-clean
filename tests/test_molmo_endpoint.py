"""Molmo HTTP endpoint regression tests without a live server."""

import os
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from capx.integrations.vision.molmo import init_molmo


class MolmoEndpointTests(unittest.TestCase):
    @staticmethod
    def response():
        response = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": '<points coords="1 0 500 500">box</points>'}}]
        }
        return response

    def test_default_endpoint_and_served_model(self):
        expected_base = os.environ.get("MOLMO_BASE_URL", "http://127.0.0.1:49188/v1")
        with patch("capx.integrations.vision.molmo.requests.Session") as session_class:
            session = session_class.return_value
            session.post.return_value = self.response()
            detector = init_molmo()
            result = detector(Image.new("RGB", (100, 80)), ["box"])
        self.assertEqual(result, {"box": (50, 40)})
        self.assertEqual(session.post.call_args.args[0], f"{expected_base.rstrip('/')}/chat/completions")
        self.assertEqual(session.post.call_args.kwargs["json"]["model"], "molmo2-8b")

    def test_explicit_endpoint_and_model_overrides_are_preserved(self):
        with patch("capx.integrations.vision.molmo.requests.Session") as session_class:
            session = session_class.return_value
            session.post.return_value = self.response()
            detector = init_molmo(
                model_name="custom-molmo", base_url="http://other-server:9000/v1/", api_key="test-key"
            )
            detector(Image.new("RGB", (100, 80)), ["box"])
        self.assertEqual(session.post.call_args.args[0], "http://other-server:9000/v1/chat/completions")
        self.assertEqual(session.post.call_args.kwargs["json"]["model"], "custom-molmo")
        self.assertEqual(session.post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")


if __name__ == "__main__":
    unittest.main()
