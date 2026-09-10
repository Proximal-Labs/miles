from typing import Annotated

import typer

MetricThresholdOption = Annotated[float, typer.Option(help="eval/gsm8k accuracy threshold")]

NumRolloutOption = Annotated[int, typer.Option(help="Number of rollouts")]

SeedOption = Annotated[int, typer.Option(help="Random seed for fault injection")]
