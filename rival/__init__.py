"""Rival: calibrated population and behavior simulation."""

from .engine import RivalEngine
from .schemas import ScenarioSpec
from .version import __version__

__all__ = ["RivalEngine", "ScenarioSpec", "__version__"]
