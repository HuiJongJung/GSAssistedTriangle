from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidualGsPolicy:
    """Static configuration for residual-Gaussian insertion.

    Mirrors the "Residual GS Policy" section of ``EXPERIMENT.md``. Defaults match
    the documented values so a bare ``ResidualGsPolicy()`` is the canonical
    experiment setting.
    """

    residual_top_percent: float = 10.0
    max_triangle_contribution: float = 0.35   # legacy alpha gate (unused by geometry gate)
    normal_top_percent: float = 15.0          # geometry gate: normal-disagreement percentile
    depth_top_percent: float = 15.0           # geometry gate: depth-instability percentile
    min_checkpoint_repeats: int = 2
    min_view_repeats: int = 3
    max_insert_per_event: int = 5000
    max_total_gs: int = 100000


def remaining_capacity(current_gs_count: int, policy: ResidualGsPolicy) -> int:
    """How many more Gaussians may exist before hitting the global cap."""
    return max(0, policy.max_total_gs - int(current_gs_count))


def can_insert_more(current_gs_count: int, policy: ResidualGsPolicy) -> bool:
    return remaining_capacity(current_gs_count, policy) > 0


def clamp_insertion_count(requested: int, current_gs_count: int,
                          policy: ResidualGsPolicy) -> int:
    """Clamp a requested insertion count to the per-event and global limits.

    Never returns a negative number; returns ``0`` once the global cap is hit.
    """
    if requested < 0:
        raise ValueError("requested insertion count must be non-negative")
    allowed = min(requested, policy.max_insert_per_event,
                  remaining_capacity(current_gs_count, policy))
    return max(0, allowed)
