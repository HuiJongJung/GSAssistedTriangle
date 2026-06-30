from __future__ import annotations


def percent_save_iterations(iterations: int, interval_percent: int = 10) -> list[int]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if interval_percent <= 0 or interval_percent > 100:
        raise ValueError("interval_percent must be in 1..100")

    points = []
    percent = interval_percent
    while percent <= 100:
        point = round(iterations * percent / 100)
        points.append(max(1, min(iterations, point)))
        percent += interval_percent
    points[-1] = iterations
    return sorted(set(points))


def default_gs_start_iter(
    start_opacity_floor: int = 5000,
    start_pruning: int = 4000,
    minimum: int = 5000,
) -> int:
    return max(minimum, start_opacity_floor, start_pruning + 1000)


def resolve_save_iterations(
    iterations: int,
    interval_percent: int = 10,
    gs_start_iter=None,
) -> list[int]:
    """Canonical checkpoint/diagnostic schedule shared by variants A/B/C.

    Saves every ``interval_percent`` of the run and, if ``gs_start_iter`` is
    provided and not already on that grid, adds it as an extra diagnostic point
    (per ``EXPERIMENT.md``). The returned list is sorted and de-duplicated.
    """
    points = percent_save_iterations(iterations, interval_percent)
    if gs_start_iter is not None and 0 < gs_start_iter <= iterations and gs_start_iter not in points:
        points = sorted(set(points + [gs_start_iter]))
    return points
