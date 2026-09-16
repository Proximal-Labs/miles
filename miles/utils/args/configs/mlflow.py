from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


# mlflow
class MlflowConfig(BaseConfig):
    use_mlflow: Annotated[bool, A("--use-mlflow", action="store_true", default=False)]
    mlflow_tracking_uri: Annotated[
        str | None,
        A(
            "--mlflow-tracking-uri",
            type=str,
            default=None,
            help="MLflow tracking server URI. Defaults to MLFLOW_TRACKING_URI env var, or local mlruns/ directory.",
        ),
    ]
    mlflow_experiment_name: Annotated[
        str,
        A("--mlflow-experiment-name", type=str, default="miles", help="MLflow experiment name."),
    ]
    mlflow_run_name: Annotated[
        str | None,
        A(
            "--mlflow-run-name",
            type=str,
            default=None,
            help="MLflow run name. Defaults to --wandb-group if not set.",
        ),
    ]
    mlflow_run_id: Annotated[str | None, A("--mlflow-run-id", type=str, default=None)]
