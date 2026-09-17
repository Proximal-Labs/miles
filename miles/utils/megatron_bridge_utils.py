from contextlib import contextmanager

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    # Float16Module casts buffers to bf16, but expert_bias must stay fp32.
    # Restore before bridge export reads the values.
    for m in model:
        for module in m.modules():
            if hasattr(module, "_maintain_float32_expert_bias"):
                module._maintain_float32_expert_bias()

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")


def apply_dsa_backend_args(provider, args) -> None:
    """Select the DSA (sparse-MLA) kernel backend on a Megatron-Bridge provider from ``--dsa-attention-backend``.

    The radixark ``bridge`` branch exposes ``dsa_attention_backend`` ("megatron" | "tilelang", vendored TileLang
    kernels). Megatron-Bridge ``main`` (radixark ``bridge-new``) drops that field and relies on Megatron-Core's
    native ``dsa_kernel_backend`` ("none" | "tilelang" | "cudnn"). Map miles' option onto whichever the provider
    has: "tilelang" -> "tilelang", "megatron" (unfused megatron-core path) -> "none". An explicit Megatron
    ``--dsa-kernel-backend`` always wins. Non-DSA providers have neither field and are left alone.
    """
    backend = getattr(args, "dsa_attention_backend", "megatron")
    if hasattr(provider, "dsa_attention_backend"):
        provider.dsa_attention_backend = backend
    elif hasattr(provider, "dsa_kernel_backend"):
        explicit = getattr(args, "dsa_kernel_backend", None)
        provider.dsa_kernel_backend = explicit or {"tilelang": "tilelang", "megatron": "none"}[backend]
