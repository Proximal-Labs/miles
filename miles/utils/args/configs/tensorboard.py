from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


# tensorboard
class TensorboardConfig(BaseConfig):
    # tb_project_name, tb_experiment_name
    use_tensorboard: Annotated[bool, A("--use-tensorboard", action="store_true", default=False)]
    tb_project_name: Annotated[
        str | None,
        A(
            "--tb-project-name",
            type=str,
            default=None,
            help="Directory to store tensorboard logs. Default is  os.environ.get('TENSORBOARD_DIR') directory.",
        ),
    ]
    tb_experiment_name: Annotated[str | None, A("--tb-experiment-name", type=str, default=None)]
