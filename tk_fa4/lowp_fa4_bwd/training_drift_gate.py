"""Pure-Python rolling loss-drift gate for matched-route training probes."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class RouteLossDriftGate:
    """Latch warnings and failures from matched-batch route loss gaps.

    The gate observes one scalar loss per route after a complete training
    round.  It only fails when the subject route's rolling mean loss exceeds
    every reference route by more than ``failure_threshold`` for
    ``failure_patience`` consecutive windows.  Consequently, route execution
    order and GPU work are unchanged until the caller elects to stop.
    """

    subject_route: str
    reference_routes: tuple[str, ...]
    window: int
    failure_threshold: float
    minimum_updates: int
    warning_threshold: float | None = None
    failure_patience: int = 1
    _gaps: dict[str, deque[float]] = field(init=False, repr=False)
    _expected_round: int = field(default=0, init=False, repr=False)
    _failure_streak: int = field(default=0, init=False, repr=False)
    _warning_active: bool = field(default=False, init=False, repr=False)
    _failure: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    _latest: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    _transitions: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.subject_route:
            raise ValueError("subject_route must not be empty")
        if not self.reference_routes:
            raise ValueError("reference_routes must not be empty")
        if len(set(self.reference_routes)) != len(self.reference_routes):
            raise ValueError("reference_routes must be unique")
        if self.subject_route in self.reference_routes:
            raise ValueError("subject_route cannot be a reference route")
        if self.window < 1:
            raise ValueError("window must be positive")
        if self.minimum_updates < self.window:
            raise ValueError("minimum_updates must be at least window")
        if self.failure_patience < 1:
            raise ValueError("failure_patience must be positive")
        if (
            not math.isfinite(self.failure_threshold)
            or self.failure_threshold <= 0.0
        ):
            raise ValueError("failure_threshold must be finite and positive")
        if self.warning_threshold is not None:
            if (
                not math.isfinite(self.warning_threshold)
                or self.warning_threshold <= 0.0
                or self.warning_threshold >= self.failure_threshold
            ):
                raise ValueError(
                    "warning_threshold must be finite, positive, and below "
                    "failure_threshold"
                )
        self._gaps = {
            route: deque(maxlen=self.window)
            for route in self.reference_routes
        }

    @property
    def warning_active(self) -> bool:
        """Whether a warning has occurred; warnings intentionally latch."""
        return self._warning_active

    @property
    def failed(self) -> bool:
        return self._failure is not None

    @property
    def failure(self) -> dict[str, Any] | None:
        return self._failure

    def observe(
        self,
        round_index: int,
        losses: Mapping[str, float],
    ) -> dict[str, Any]:
        """Observe one complete matched-route round and return its report."""
        if self.failed:
            raise RuntimeError("cannot observe after the drift gate has failed")
        if round_index != self._expected_round:
            raise ValueError(
                f"expected round {self._expected_round}, got {round_index}"
            )
        required_routes = (self.subject_route, *self.reference_routes)
        missing = [route for route in required_routes if route not in losses]
        if missing:
            raise KeyError(f"missing route losses: {missing}")
        values = {route: float(losses[route]) for route in required_routes}
        nonfinite = [
            route for route, value in values.items() if not math.isfinite(value)
        ]
        if nonfinite:
            raise ValueError(f"non-finite route losses: {nonfinite}")

        subject_loss = values[self.subject_route]
        for route in self.reference_routes:
            self._gaps[route].append(subject_loss - values[route])
        self._expected_round += 1
        completed_updates = round_index + 1
        ready = (
            completed_updates >= self.minimum_updates
            and all(len(gaps) == self.window for gaps in self._gaps.values())
        )
        rolling_gaps = (
            {
                route: statistics.fmean(gaps)
                for route, gaps in self._gaps.items()
            }
            if ready
            else None
        )
        warning_exceeded = bool(
            ready
            and self.warning_threshold is not None
            and all(
                gap > self.warning_threshold
                for gap in rolling_gaps.values()
            )
        )
        failure_exceeded = bool(
            ready
            and all(
                gap > self.failure_threshold
                for gap in rolling_gaps.values()
            )
        )
        self._failure_streak = (
            self._failure_streak + 1 if failure_exceeded else 0
        )
        report = {
            "round": round_index,
            "completed_updates": completed_updates,
            "ready": ready,
            "rolling_mean_loss_gaps": rolling_gaps,
            "warning_exceeded": warning_exceeded,
            "failure_exceeded": failure_exceeded,
            "failure_streak": self._failure_streak,
            "warning_active": self._warning_active,
            "failed": False,
        }
        if warning_exceeded and not self._warning_active:
            self._warning_active = True
            report["warning_active"] = True
            self._transitions.append({"kind": "warning", **report})
        if self._failure_streak >= self.failure_patience:
            report["failed"] = True
            self._failure = report.copy()
            self._transitions.append({"kind": "failure", **report})
        self._latest = report
        return report

    def as_dict(self) -> dict[str, Any]:
        """Return a compact JSON-serializable checkpoint of gate state."""
        return {
            "configuration": {
                "subject_route": self.subject_route,
                "reference_routes": list(self.reference_routes),
                "window": self.window,
                "warning_threshold": self.warning_threshold,
                "failure_threshold": self.failure_threshold,
                "failure_patience": self.failure_patience,
                "minimum_updates": self.minimum_updates,
                "comparison": (
                    "subject rolling mean loss minus each matched-batch "
                    "reference"
                ),
                "failure_rule": "all reference gaps strictly exceed threshold",
            },
            "observed_rounds": self._expected_round,
            "warning_active": self.warning_active,
            "failure_streak": self._failure_streak,
            "failed": self.failed,
            "latest": self._latest,
            "failure": self.failure,
            "transitions": self._transitions,
        }
