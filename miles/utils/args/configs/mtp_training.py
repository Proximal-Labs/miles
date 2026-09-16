from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


class MtpTrainingConfig(BaseConfig):
    """Add MTP training specific arguments."""

    mtp_num_layers: Annotated[int | None, A("--mtp-num-layers", reset=True, type=int, default=None)]
    mtp_loss_scaling_factor: Annotated[float, A("--mtp-loss-scaling-factor", reset=True, type=float, default=0.2)]
    enable_mtp_training: Annotated[
        bool,
        A(
            "--enable-mtp-training",
            action="store_true",
            default=False,
            help="Enable MTP layer parameter updates during training",
        ),
    ]
