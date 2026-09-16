"""Native-tool integration checks and policy-loop failure handling."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import workplace_agent
from fastapi.testclient import TestClient
from prepare_workplace import convert
from workplace_server import create_app


class NativeWorkplaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path(os.environ["WORKPLACE_TEST_DATASET"])
        rows = [json.loads(line) for line in cls.source.read_text().splitlines()]
        cls.rows = list({row["category"]: row for row in rows}.values())

    def test_gold_noop_isolation_and_omitted_action(self) -> None:
        with TestClient(create_app(self.source)) as client:
            for row in self.rows:
                first = client.post(f'/sessions/{row["id"]}').json()["session_id"]
                second = client.post(f'/sessions/{row["id"]}').json()["session_id"]
                for call in row["ground_truth"]:
                    result = client.post(f"/sessions/{first}/tool", json=call)
                    self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(client.post(f"/sessions/{first}/verify").json()["reward"], 1)
                self.assertEqual(client.post(f"/sessions/{second}/verify").json()["reward"], 0)
                third = client.post(f'/sessions/{row["id"]}').json()["session_id"]
                for call in row["ground_truth"][:-1]:
                    client.post(f"/sessions/{third}/tool", json=call).raise_for_status()
                self.assertEqual(client.post(f"/sessions/{third}/verify").json()["reward"], 0)
            self.assertEqual(client.get("/health").json()["active"], 0)

    def test_bad_arguments_visible_and_verifier_failure_not_reward(self) -> None:
        with TestClient(create_app(self.source), raise_server_exceptions=False) as client:
            sid = client.post(f'/sessions/{self.rows[0]["id"]}').json()["session_id"]
            body = {"name": self.rows[0]["ground_truth"][0]["name"], "arguments": "[]"}
            self.assertIn("Error executing tool", client.post(f"/sessions/{sid}/tool", json=body).json()["output"])
            with patch("workplace_server.is_correct", side_effect=RuntimeError("verifier unavailable")):
                response = client.post(f"/sessions/{sid}/verify")
            self.assertEqual(response.status_code, 500)
            self.assertEqual(client.get("/health").json()["active"], 0)
            self.assertEqual(client.post("/sessions/-1").status_code, 404)

    def test_export_has_no_gold_or_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "train.jsonl"
            receipt = convert(self.source, target)
            exported = [json.loads(line) for line in target.read_text().splitlines()]
            native = [json.loads(line) for line in self.source.read_text().splitlines()]
            self.assertEqual(receipt["tasks"], len(native))
            for row, original in zip(exported, native, strict=True):
                self.assertEqual(row["prompt"], original["responses_create_params"]["input"])
                self.assertEqual(row["metadata"]["workplace_policy"], original["responses_create_params"])
                self.assertNotIn("ground_truth", row["metadata"])
                self.assertNotIn("provenance", row["metadata"])


class FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs) -> list[int]:
        return [1, 2]


class PolicyLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_turns_preserve_reasoning_and_native_reward(self) -> None:
        requests = []
        count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal count
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if request.url.path.endswith("/chat/completions"):
                count += 1
                message = {"content": "done", "reasoning_content": "reasoning retained"}
                reason = "stop"
                if count == 1:
                    reason = "tool_calls"
                    message["tool_calls"] = [
                        {"id": "call_1", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}
                    ]
                return httpx.Response(
                    200,
                    json={
                        "choices": [{"finish_reason": reason, "message": message}],
                        "usage": {"completion_tokens": 5, "prompt_tokens": 2},
                    },
                )
            if request.url.path.endswith("/tool"):
                return httpx.Response(200, json={"output": "tool observation"})
            return httpx.Response(
                200, json={"valid": True, "state_replay_consistent": True, "reward": 1.0, "tool_calls": 1}
            )

        params = {
            "input": [{"role": "user", "content": "task"}],
            "tools": [{"type": "function", "name": "test_tool", "parameters": {}}],
        }
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch("workplace_agent.tokenizer", return_value=FakeTokenizer()):
                result = await workplace_agent.policy_loop(
                    client,
                    "http://policy/sessions/x",
                    "http://resource/sessions/y",
                    params,
                    {"max_tokens": 1000},
                    81920,
                    24,
                )
        self.assertTrue(result["workplace_episode_valid"])
        self.assertEqual(result["workplace_reward"], 1.0)
        self.assertEqual(result["workplace_turns"], 2)
        policy = [body for path, body in requests if path.endswith("/chat/completions")]
        self.assertEqual(policy[1]["messages"][1]["reasoning_content"], "reasoning retained")
        self.assertEqual(policy[1]["max_tokens"], 995)
        self.assertEqual(json.loads(policy[1]["messages"][2]["content"]), {"output": "tool observation"})

    async def test_context_retry_uses_exact_server_count(self) -> None:
        budgets = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            budgets.append(body["max_tokens"])
            if len(budgets) == 1:
                return httpx.Response(400, json={"message": "30000 tokens from the input messages"})
            return httpx.Response(200, json={"ok": True})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await workplace_agent.completion(client, "http://policy", {"max_tokens": 40000}, 50000)
        self.assertTrue(response["ok"])
        self.assertEqual(budgets, [40000, 19992])

    async def test_abort_and_http_failure_never_become_zero_reward(self) -> None:
        params = {"input": [{"role": "user", "content": "task"}], "tools": []}
        for code in (200, 503):

            def handler(request: httpx.Request, status_code: int = code) -> httpx.Response:
                return httpx.Response(status_code, json={"choices": [{"finish_reason": "abort"}]})

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with patch("workplace_agent.tokenizer", return_value=FakeTokenizer()):
                    call = workplace_agent.policy_loop(
                        client, "http://policy", "http://resource", params, {"max_tokens": 1000}, 81920, 24
                    )
                    if code == 503:
                        with self.assertRaises(httpx.HTTPStatusError):
                            await call
                    else:
                        self.assertFalse((await call)["workplace_episode_valid"])


if __name__ == "__main__":
    unittest.main()
