"""One parent and two independent reviewers in a single v2 training episode."""

import anyio

import httpx

from miles.rollout.agentic.harness import AgentResult, GenerationRequest
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
        parent = SessionContext(agent_run_id="main", context_id="main")
        draft = await _complete(client, base_url, parent, messages, request_kwargs)
        draft_message = draft["choices"][0]["message"]
        reviews = {}
        reviewer_names = ("review-a", "review-b")

        async def review(name):
            context = SessionContext(agent_run_id=name, context_id=name, parent_agent_run_id="main")
            response = await _complete(
                client,
                base_url,
                context,
                [{"role": "user", "content": f"Review this answer independently: {draft_message['content']}"}],
                request_kwargs,
            )
            reviews[name] = response["choices"][0]["message"]["content"]

        async with anyio.create_task_group() as group:
            for name in reviewer_names:
                group.start_soon(review, name)
        critiques = [reviews[name] for name in reviewer_names]
        messages += [draft_message, {"role": "user", "content": f"Revise using these reviews: {critiques}"}]
        final = await _complete(client, base_url, parent, messages, request_kwargs, previous_response_id=draft["id"])
    return AgentResult(metadata={"answer": final["choices"][0]["message"]["content"]}, producer_finished=True)
