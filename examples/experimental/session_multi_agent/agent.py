"""One parent and two independent reviewers in a single v2 training episode."""

import asyncio

import httpx

from miles.rollout.agentic.harness import AgentResult, AgentRun, GenerationRequest
from miles.rollout.session.v2.contexts import SessionContext


async def _complete(client, base_url, context, messages, request_kwargs, *, previous_response_id=None):
    generation = GenerationRequest(context, previous_response_id=previous_response_id)
    response = await client.post(
        f"{base_url}/v1/chat/completions",
        json={**request_kwargs, "messages": messages},
        headers=generation.headers(),
    )
    response.raise_for_status()
    return response.json()


async def run(base_url, prompt, request_kwargs, metadata, **kwargs) -> AgentResult:
    messages = list(prompt) if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
    async with httpx.AsyncClient(timeout=120) as client:
        run = AgentRun(base_url, client)
        async with run:
            parent = await run.register_context(SessionContext(agent_run_id="main", context_id="main"))
            draft = await _complete(client, base_url, parent, messages, request_kwargs)
            draft_message = draft["choices"][0]["message"]
            reviews = []
            for name in ("review-a", "review-b"):
                context = await run.register_context(
                    SessionContext(agent_run_id=name, context_id=name, parent_agent_run_id="main")
                )
                reviews.append(
                    run.create_task(
                        _complete(
                            client,
                            base_url,
                            context,
                            [
                                {
                                    "role": "user",
                                    "content": f"Review this answer independently: {draft_message['content']}",
                                }
                            ],
                            request_kwargs,
                        )
                    )
                )
            responses = await asyncio.gather(*reviews)
            critiques = [response["choices"][0]["message"]["content"] for response in responses]
            messages += [draft_message, {"role": "user", "content": f"Revise using these reviews: {critiques}"}]
            final = await _complete(
                client, base_url, parent, messages, request_kwargs, previous_response_id=draft["id"]
            )
        return run.result({"answer": final["choices"][0]["message"]["content"]})
