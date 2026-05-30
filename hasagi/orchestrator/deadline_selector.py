"""Mid-run deadline floor: re-evaluate the MSS each tick.

The admission helper (``energy_admit_or_drop``) checks the deadline + energy
budget exactly once at submission. After that the scheduler is free to
throttle or scale down to save energy, but if the job's remaining work no
longer fits its remaining wall-clock budget, an energy-only policy has no
way to notice.

``DeadlineFloorSelector`` plugs that gap. Each tick it computes the
*deadline floor* — the smallest GPU count that still finishes the
remaining iterations before the deadline — and exposes it to the control
loop so the loop can override any energy decision that drops below the
floor. When the job is already infeasible (deadline missed, no allocation
meets it), the selector returns ``max_gpus`` so the loop hands the job
every GPU available and the operator can act on the resulting failure.

Independent of the energy axis: the floor only encodes "you must run this
fast or faster"; it never asks for *more* GPUs than the deadline needs.
The energy policy is free to pick anything ``>= floor``.
"""
from __future__ import annotations

from dataclasses import dataclass

from hasagi.admission.mss import ScalingCurve, minimum_satisfactory_share


@dataclass(frozen=True)
class DeadlineFloor:
    """Result of one floor evaluation."""

    gpus: int
    feasible: bool
    iters_remaining: int
    deadline_seconds_remaining: float
    reason: str


@dataclass
class DeadlineFloorSelector:
    """Per-tick gpu-floor from remaining-iters + remaining-deadline.

    Args:
        curve: throughput scaling curve (iter/s per GPU). Same object the
            admission helper used; the floor inherits its concavity.
        min_gpus: lower clamp so an idle / nearly-finished job still keeps
            a baseline allocation. Default 1.
        infeasible_floor: GPU count returned when no allocation meets the
            deadline. Default ``curve.max_gpus`` — the loop hands the job
            the cluster's headroom and surfaces the miss via reason.
    """

    curve: ScalingCurve
    min_gpus: int = 1
    infeasible_floor: int | None = None

    def evaluate(
        self,
        iters_remaining: int,
        deadline_seconds_remaining: float,
    ) -> DeadlineFloor:
        max_gpus = self.curve.max_gpus
        ceiling = self.infeasible_floor if self.infeasible_floor is not None else max_gpus

        if iters_remaining <= 0:
            return DeadlineFloor(
                gpus=self.min_gpus,
                feasible=True,
                iters_remaining=iters_remaining,
                deadline_seconds_remaining=deadline_seconds_remaining,
                reason="no iters remaining — floor at min",
            )

        if deadline_seconds_remaining <= 0:
            return DeadlineFloor(
                gpus=min(ceiling, max_gpus),
                feasible=False,
                iters_remaining=iters_remaining,
                deadline_seconds_remaining=deadline_seconds_remaining,
                reason="deadline already passed — escalate to ceiling",
            )

        mss = minimum_satisfactory_share(
            iterations_remaining=iters_remaining,
            deadline_seconds=deadline_seconds_remaining,
            curve=self.curve,
        )
        if mss == 0:
            return DeadlineFloor(
                gpus=min(ceiling, max_gpus),
                feasible=False,
                iters_remaining=iters_remaining,
                deadline_seconds_remaining=deadline_seconds_remaining,
                reason=(
                    f"infeasible: {iters_remaining} iters in {deadline_seconds_remaining:.1f}s "
                    f"unreachable at max throughput {self.curve.throughput(max_gpus):.3f} iter/s"
                ),
            )

        floor = max(self.min_gpus, mss)
        return DeadlineFloor(
            gpus=floor,
            feasible=True,
            iters_remaining=iters_remaining,
            deadline_seconds_remaining=deadline_seconds_remaining,
            reason=(
                f"MSS={mss} meets {iters_remaining} iters in "
                f"{deadline_seconds_remaining:.1f}s (floor={floor})"
            ),
        )
