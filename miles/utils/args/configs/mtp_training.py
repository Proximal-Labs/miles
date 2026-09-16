from miles.utils.args.schema import A, Arg, BaseConfig


class MtpTrainingConfig(BaseConfig):
    """Add MTP training specific arguments."""

    mtp_num_layers: A[int | None, Arg(reset=True)] = None
    mtp_loss_scaling_factor: A[float, Arg(reset=True)] = 0.2
    enable_mtp_training: A[bool, Arg(help="Enable MTP layer parameter updates during training")] = False
