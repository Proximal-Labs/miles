# CPU-only test environment. No model weights, GPU runtime, or credentials.
FROM python:3.12-slim
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir triton==3.6.0 'pydantic>=2.10,<3' 'httpx>=0.28,<1' modal==1.5.5 \
    pytest pytest-asyncio fastapi uvicorn numpy safetensors transformers==5.12.1 \
    pybase64 pillow jinja2 'ray[default]' aiohttp pyyaml omegaconf openai==2.6.1 \
    orjson pyzmq msgspec sentencepiece tiktoken partial_json_parser setproctitle psutil \
    einops compressed-tensors gguf scipy sglang-router datasets wandb tensorboard blake3 xxhash zstandard \
    torchao ipython xgrammar polars qwen_vl_utils kubernetes_asyncio mcp blobfile pylatexenc \
    mypy ruff==0.14.7 black==24.3.0 isort==5.13.2
ENV PYTHONPATH=/work:/sglang/python PYTHONDONTWRITEBYTECODE=1 PROXIMAL_TEST_TOKENIZER=/tokenizer
WORKDIR /work
