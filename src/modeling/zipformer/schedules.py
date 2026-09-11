from .scaling import ScheduledFloat


def _whitening_schedule(x: float, ratio: float = 2.0) -> ScheduledFloat:
    return ScheduledFloat((0.0, x), (20000.0, ratio * x), default=x)
