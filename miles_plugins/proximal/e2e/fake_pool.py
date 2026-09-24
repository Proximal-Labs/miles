"""A local stand-in for the Miles-owned serving pool, for the offline Stage A proof.

It honors the gateway's contract (see gateway.py): ``/policies/prepare`` returns
verification evidence for any requested immutable version, and
``/v1/chat/completions`` requires the version header plus the matching adapter model
name, then answers with SGLang-shaped exact token IDs and logprobs from the real
tokenizer. It calls the bash tool once (tool arguments with a space, so agent-px's
compact re-serialization is exercised), then submits with mini-swe's command. Swap the run
config's ``inference_url`` to the real pool to test real SGLang and LoRA loading.

    python -m miles_plugins.proximal.e2e.fake_pool --config run.json --port 9012
"""

import argparse
import hmac
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from miles_plugins.proximal.contracts import RunConfig, read_run_config
from miles_plugins.proximal.gateway import PreparePolicy

TOOL_TEXT = (
    'Let me look.</think>\n\n<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call><|im_end|>'
)
SUBMIT_TEXT = (
    'Implemented.</think>\n\n<tool_call>\n{"name": "bash", "arguments": '
    '{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}}\n</tool_call><|im_end|>'
)


class FakePool:
    def __init__(self, run: RunConfig, *, tokenizer: object, api_key: str):
        self.run, self.tokenizer, self.api_key = run, tokenizer, api_key
        self.app = FastAPI()
        self.requests = 0
        self._routes()

    def _authorize(self, request: Request) -> None:
        if not hmac.compare_digest(request.headers.get("authorization", ""), self.api_key):
            raise HTTPException(401, "Invalid serving pool credential")

    def _request_model(self, sha256: str) -> str:
        return f"{self.run.base_model.name}:miles-{sha256}"

    def _reply(self, input_ids: list[int]) -> tuple[dict[str, object], str, list[int]]:
        prompt = self.tokenizer.decode(input_ids)  # type: ignore[attr-defined]
        if "<tool_response>" not in prompt:
            message: dict[str, object] = {
                "role": "assistant",
                "reasoning_content": "Let me look.",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                    }
                ],
            }
            return message, "tool_calls", self.tokenizer.encode(TOOL_TEXT, add_special_tokens=False)  # type: ignore[attr-defined]
        message = {
            "role": "assistant",
            "reasoning_content": "Implemented.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}',
                    },
                }
            ],
        }
        return message, "tool_calls", self.tokenizer.encode(SUBMIT_TEXT, add_special_tokens=False)  # type: ignore[attr-defined]

    def _routes(self) -> None:
        @self.app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @self.app.post("/policies/prepare")
        async def prepare(body: PreparePolicy, request: Request) -> dict[str, object]:
            self._authorize(request)
            if body.base_model != self.run.base_model:
                raise HTTPException(409, "Requested base differs from the pool")
            return {
                "snapshot": body.snapshot.model_dump(),
                "base_model": body.base_model.model_dump(),
                "request_model": self._request_model(body.snapshot.sha256),
            }

        @self.app.post("/v1/chat/completions")
        async def chat(request: Request) -> JSONResponse:
            self._authorize(request)
            sha256 = request.headers.get("x-proximal-policy-sha256", "")
            body = await request.json()
            # As strict as gateway.py: explicitly non-streaming.
            if body.get("model") != self._request_model(sha256) or body.get("stream") is not False:
                raise HTTPException(409, "Request must name the verified adapter, non-streaming")
            input_ids = body["input_ids"]
            message, finish_reason, ids = self._reply(input_ids)
            self.requests += 1
            return JSONResponse(
                headers={
                    "x-proximal-policy-sha256": sha256,
                    "x-proximal-base-revision": self.run.base_model.revision,
                },
                content={
                    "id": f"chat-{self.requests}",
                    "created": 1,
                    "object": "chat.completion",
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": finish_reason,
                            "message": message,
                            "meta_info": {
                                "output_token_logprobs": [[-0.25, token, None] for token in ids],
                                "prompt_tokens": len(input_ids),
                                "completion_tokens": len(ids),
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(input_ids),
                        "completion_tokens": len(ids),
                        "total_tokens": len(input_ids) + len(ids),
                    },
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9012)
    args = parser.parse_args()
    run = read_run_config(args.config)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(run.tokenizer_path), local_files_only=True)
    [auth_env] = [env for name, env in run.inference_header_env.items() if name.lower() == "authorization"]
    pool = FakePool(run, tokenizer=tokenizer, api_key=os.environ[auth_env])
    import uvicorn

    uvicorn.run(pool.app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
