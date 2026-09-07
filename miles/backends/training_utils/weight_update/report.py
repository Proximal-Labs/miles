from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class WeightUpdateReport:
    weight_version: int | None
    updated_cell_ids: tuple[str, ...]

    def validate_assignment(self, assigned_cell_ids: Sequence[str]) -> None:
        assigned = frozenset(assigned_cell_ids)
        assert len(assigned) == len(assigned_cell_ids), f"a cell is assigned twice, got {list(assigned_cell_ids)}"

        unknown = sorted(frozenset(self.updated_cell_ids) - assigned)
        assert not unknown, f"cells {unknown} were never assigned to this trainer, which owns {sorted(assigned)}"

    @classmethod
    def combine(cls, reports: Sequence["WeightUpdateReport"]) -> "WeightUpdateReport":
        assert reports, "no trainer cell reported the outcome of this update"

        versions = {report.weight_version for report in reports if report.weight_version is not None}
        assert len(versions) <= 1, f"the trainer cells published different weight versions, got {sorted(versions)}"

        return cls(
            weight_version=next(iter(versions), None),
            updated_cell_ids=tuple(cell_id for report in reports for cell_id in report.updated_cell_ids),
        )
