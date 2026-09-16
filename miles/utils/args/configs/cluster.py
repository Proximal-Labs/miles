import argparse
from typing import Annotated

from miles.utils.args.schema import A, BaseConfig
from miles.utils.workers.types import ClusterBackend, DeployComponent, WorkerCommBackend


# Ray
class ClusterConfig(BaseConfig):
    cluster_backend: Annotated[str, A(
        "--cluster-backend", type=str, default=ClusterBackend.RAY.value,
        choices=tuple(backend.value for backend in ClusterBackend),
        help=(
            "Which backend provides the worker processes: "
            "`ray` launches them from the driver, `kubernetes` expects the platform to have "
            "created them already and observes them by their pod labels."
        ),
    )]
    worker_comm_backend: Annotated[str | None, A(
        "--worker-comm-backend", type=str, default=None,
        choices=tuple(backend.value for backend in WorkerCommBackend),
        help=(
            "How the driver calls its workers: `ray` sends actor calls, `rpc` calls the http server "
            "every worker serves. Unset picks the default of the cluster backend, today `ray` under "
            "`--cluster-backend ray` and `rpc` under `--cluster-backend kubernetes`."
        ),
    )]
    deploy_component: Annotated[str, A(
        "--deploy-component", type=str, default=DeployComponent.ALL.value,
        choices=tuple(component.value for component in DeployComponent),
        help=(
            "Which part of the run this launch deploys: `all` deploys every worker, `trainer` the trainer "
            "controllers and their megatron ranks, `inference` a group of inference engines that registers "
            "itself into the run, and `primary` everything else (orchestration script, rollout executor, "
            "session servers, inference controller and routers). Deploying a subset takes one launch per "
            "subset, and the launch that carries the orchestration script reaches the trainer through the "
            "addresses it is given."
        ),
    )]
    deploy_instance_id: Annotated[str | None, A(
        "--deploy-instance-id", type=str, default=None,
        help=(
            "Id of this deployment, telling it apart from the other deployments of the same component "
            "in the same run: a trainer id such as `trainer-a` under `--deploy-component trainer`, or an "
            "engine group id such as `inf-east` under `--deploy-component inference`. A deployment's "
            "arguments describe only what it carries, so this id selects nothing; it is required under "
            "`--deploy-component inference`, which names its engine pools by it, optional under "
            "`--deploy-component trainer`, whose config already declares the one trainer id it carries, and "
            "refused for `all` and `primary`, which a run has exactly one of."
        ),
    )]
    init_expected_num_cells: Annotated[int | None, A(
        "--init-expected-num-cells", type=int, default=None,
        help=(
            "How many engine cells per model this run waits for before it starts, when the engines are "
            "deployed elsewhere and register themselves into it. The run cannot derive the number, because "
            "the engine deployments are launched separately and may arrive late; declare here how many "
            "cells the first rollout needs. It gates startup only, and the run keeps serving whatever "
            "registers or leaves afterwards."
        ),
    )]
    trainer_controller_addrs: Annotated[list[str] | None, A(
        "--trainer-controller-addrs", type=str, default=None, nargs="+",
        help=(
            "Address of every independently deployed trainer controller, one "
            "<trainer_id>=<host:port> entry per trainer the run drives. Required when this launch "
            "carries the orchestration script but not the trainer."
        ),
    )]
    inference_controller_addr: Annotated[str | None, A(
        "--inference-controller-addr", type=str, default=None,
        help=(
            "Address of the one inference controller of the run, as host:port. Given "
            "to a `--deploy-component inference` launch, whose reporter registers the engines it deploys "
            "into that controller."
        ),
    )]
    actor_num_nodes: Annotated[int, A("--actor-num-nodes", type=int, default=1, help="Number of nodes for training actor")]
    actor_num_gpus_per_node: Annotated[int, A(
        "--actor-num-gpus-per-node", type=int, default=8, help="Number of gpus per node for training actor",
    )]
    critic_num_nodes: Annotated[int | None, A(
        "--critic-num-nodes", type=int, default=None, help="Number of nodes for training actor",
    )]
    critic_num_gpus_per_node: Annotated[int | None, A(
        "--critic-num-gpus-per-node", type=int, default=None, help="Number of gpus per node for training actor",
    )]
    rollout_num_gpus: Annotated[int | None, A(
        "--rollout-num-gpus", type=int, default=None,
        help=(
            "Number of GPUs for inference. Note that when using --colocate, "
            "i.e. the training and the inference engines are on the same gpus, this param will be ignored and will be set as "
            "actor_num_gpus_per_node * actor_num_nodes."
        ),
    )]
    rollout_num_gpus_per_engine: Annotated[int, A(
        "--rollout-num-gpus-per-engine", type=int, default=1,
        help="Number of GPUs per inference engine, just like the tp_size in sglang.",
    )]
    num_gpus_per_node: Annotated[int, A(
        "--num-gpus-per-node", type=int, default=8,
        help=(
            "Number of gpus per node for rollout."
            "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
        ),
    )]
    colocate: Annotated[bool, A(
        "--colocate", action="store_true", default=False,
        help=(
            "Whether to colocate the inference engines and the actor. "
            "Turning this on will also set --offload to true."
        ),
    )]
    offload: Annotated[bool, A(
        "--offload", action="store_true", default=False, help=("Equivalent to --offload-train + --offload-rollout. "),
    )]
    offload_train: Annotated[bool | None, A(
        "--offload-train", action=argparse.BooleanOptionalAction,
        help=(
            "Whether to offload the training actor to CPU while the rollout engines generate. "
            "Defaults to true when --colocate is set; an explicit --no-offload-train is respected."
        ),
    )]
    clear_quantized_weight_workspaces_on_offload: Annotated[bool, A(
        "--clear-quantized-weight-workspaces-on-offload", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Drop TransformerEngine's cached quantized weights before offloading the "
            "training actor. They are rebuilt on the next forward, so backing them up "
            "to pinned host memory is pure overhead. Ignored when TransformerEngine "
            "is not in use or CUDA graphs are enabled."
        ),
    )]
    offload_rollout: Annotated[bool | None, A(
        "--offload-rollout", action=argparse.BooleanOptionalAction,
        help=(
            "Whether to offload the rollout generator to CPU during training. "
            "Defaults to true when --colocate is set; an explicit --no-offload-rollout is respected."
        ),
    )]
    offload_rollout_level: Annotated[list[str], A(
        "--offload-rollout-level", type=str, nargs="+", default=["kv_cache", "weight"],
        help=(
            "Specifies what to offload during rollout when offload-rollout is set. "
            "Possible values: 'kv_cache', 'weight'. Default: both 'kv_cache' and 'weight'. "
            "Example: --offload-rollout-level kv_cache weight"
        ),
    )]
    offload_train_target: Annotated[str, A(
        "--offload-train-target", type=str, choices=["cpu", "disk"], default="cpu",
        help=(
            "Where the training actor is backed up while offloaded during rollout "
            "(only used with --offload-train on the megatron backend). "
            "'cpu' (default) keeps a pinned host copy; 'disk' streams it to node-local "
            "NVMe (--offload-train-disk-dir) for the case where even CPU RAM cannot hold it."
        ),
    )]
    stream_optimizer_state_to_disk: Annotated[bool, A(
        "--stream-optimizer-state-to-disk", action="store_true",
        help=(
            "Hold optimizer state in files on node-local NVMe, for when it does not fit the "
            "GPU *while the step runs*; --offload-train-target=disk cannot help there.\n"
            "adam: streams fp32 main params and moments through per-bucket files, one bucket "
            "resident at a time. Requires the distributed optimizer, excludes "
            "--offload-optimizer-states and --optimizer-cpu-offload.\n"
            "dist_muon: the disk backend for --chunked-optimizer-state-offload, so pass that "
            "plus a non-zero --optimizer-state-offload-fraction. --optimizer-cpu-offload is "
            "Adam-only. This bounds host residency, not the GPU restore window -- for that "
            "set --optimizer-state-offload-chunk-size-mb, which Megatron warns about at 0."
        ),
    )]
    stream_optimizer_state_moment_dtype: Annotated[str, A(
        "--stream-optimizer-state-moment-dtype", type=str, default="fp32",
        choices=["fp32", "bf16", "fp16", "fp8e4m3", "fp8e5m2"],
        help=(
            "On-disk dtype for the streamed Adam moments; the fp32 master copy is always "
            "fp32. This is a serialization format, not a compute precision: the step still "
            "hands fp32 tensors to FusedAdam, and the cast happens on the way to and from "
            "disk. That makes it distinct from --exp-avg-dtype / --exp-avg-sq-dtype, which "
            "change what the optimizer holds and require --use-precision-aware-optimizer, "
            "so they cannot be combined with streaming at all. "
            "The step is I/O bound and the moments tolerate less precision than the master "
            "copy, so bf16 cuts streaming volume by a third (12 bytes per param to 8). "
            "fp32 is bit-identical to keeping the moments on GPU. The fp8 options need "
            "per-block scaling for exp_avg_sq to be sound, which this does not implement, "
            "and are not recommended."
        ),
    )]
    offload_train_disk_dir: Annotated[str | None, A(
        "--offload-train-disk-dir", type=str, default=None,
        help=(
            "Node-local directory for the disk-offload files, used by both "
            "--offload-train-target=disk and --stream-optimizer-state-to-disk (each under "
            "its own subdirectory). Should be fast local NVMe (e.g. /scratch); a tmpfs "
            "mount, which /tmp is on many systems, keeps the data in RAM and defeats both. "
            "Files are per-process and overwritten in place every step (bounded size); "
            "defaults to $SCRATCH/miles_train_offload_<uid>. Muon's optimizer-state buffers "
            "are unlinked once mapped, so their footprint shows in df but not du."
        ),
    )]
    offload_train_disk_chunk_mb: Annotated[int, A(
        "--offload-train-disk-chunk-mb", type=int, default=256,
        help=(
            "Chunk size (MiB) for the GPU<->disk transfers, i.e. the pinned host staging "
            "buffer, which bounds host memory regardless of how much is moved. Used by both "
            "--offload-train-target=disk and --stream-optimizer-state-to-disk, and each "
            "allocates its own, so enabling both costs 2x this per rank."
        ),
    )]
    colocate_memory_peak_device: Annotated[str, A(
        "--colocate-memory-peak-device", type=str, choices=["cpu", "gpu"], default="cpu",
        help=(
            "Which device absorbs the trainer<->rollout handoff overlap. 'cpu' "
            "(default): each side offloads before the other onloads, so the "
            "engine's weight mirror and the trainer's backup briefly coexist in "
            "host memory. 'gpu': onload the other side first, so both sides "
            "briefly coexist in GPU memory instead and the two host copies never "
            "overlap. Use 'gpu' when host RAM is the tighter budget than the "
            "handoff headroom on the GPU."
        ),
    )]
    distributed_backend: Annotated[str, A("--distributed-backend", reset=True, type=str, default="nccl")]
    distributed_timeout_minutes: Annotated[int, A("--distributed-timeout-minutes", reset=True, type=int, default=10)]
