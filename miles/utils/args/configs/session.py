from typing import Annotated

from miles.utils.args.schema import A, BaseConfig
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType


class SessionConfig(BaseConfig):
    use_session_server: Annotated[
        str | bool,
        A(
            "--use-session-server",
            nargs="?",
            const=True,
            default=False,
            help="Start a standalone session server for TITO/session support. "
            "Requires --hf-checkpoint. A named --tito-model resolves its registered template; "
            "--tito-model=default uses the checkpoint-native or explicit --chat-template-path template. "
            "Bare flag (or 'v1') selects the append-only linear v1 server; "
            "'--use-session-server v2' selects the tree-serving v2 "
            "(multi-lineage trajectories, always-branch).",
        ),
    ]
    session_server_workers: Annotated[
        int, A("--session-server-workers", type=int, default=32, help="Number of session server instances.")
    ]
    session_server_ip: Annotated[
        str | None,
        A(
            "--session-server-ip",
            type=str,
            default=None,
            help="Address the session servers bind to, e.g. 0.0.0.0 to accept traffic from outside "
            "the cluster. Peers still reach them on the address their worker was placed on. "
            "Defaults to that placed address.",
        ),
    ]
    session_server_port: Annotated[
        int | None,
        A(
            "--session-server-port",
            type=int,
            default=None,
            help="Base port for the session servers, so a network policy can whitelist a known range. "
            "Instance i listens on this port plus i. Defaults to a dynamically allocated port.",
        ),
    ]
    tito_model: Annotated[
        str,
        A(
            "--tito-model",
            type=str,
            default="default",
            choices=[t.value for t in TITOTokenizerType],
            help="TITO tokenizer type for pretokenized prefix reuse. "
            "Controls how token IDs are computed for messages appended after "
            "the pretokenized prefix in multi-turn agentic sessions.",
        ),
    ]
    session_message_matcher: Annotated[
        str,
        A(
            "--session-message-matcher",
            type=str,
            default="strict",
            help=(
                "Process-wide session history matcher: strict (default), "
                "loose_tool_call, role_content_only, or a trusted dotted import "
                "path. role_content_only is a high-risk opt-in that can collapse "
                "different tool-call lineages and does not reconcile call IDs."
            ),
        ),
    ]
    session_sample_picker_path: Annotated[
        str,
        A(
            "--session-sample-picker-path",
            type=str,
            default="miles.rollout.session.v2.picker_hub.drop_retries",
            help="v2 only. Import path of the sample-pick hook for the "
            "session samples op: fn(leaf_samples, session_metadata) -> "
            "list[Sample], a pure selection over the per-leaf raw samples. "
            "Runs synchronously inside the session server process; long CPU "
            "work stalls every session on the instance. Default: the "
            "temporal-supersession retry trim.",
        ),
    ]
    session_sample_postprocessor_path: Annotated[
        str,
        A(
            "--session-sample-postprocessor-path",
            type=str,
            default="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
            help="v2 only. Import path of the post-process hook for the "
            "session samples op: fn(leaf_samples, session_metadata) -> "
            "list[Sample], finalizing loss masks / rewards over the picked "
            "samples. Runs synchronously inside the session server process. "
            "Default: exactly-once completion masking + rewards keyed by "
            "response id.",
        ),
    ]
