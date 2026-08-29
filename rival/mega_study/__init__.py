"""Prospective Twin-2K-500 Mega-Study benchmark for genuinely new situations.

This package is deliberately separate from :mod:`rival.live_pilot`.  The
existing Wave-4 benchmark is a retest study; this package evaluates transfer
to later, previously unseen Mega-Study surveys.
"""

from .constants import SCHEMA_VERSION, STUDY_ID, VARIANTS

__all__ = ["SCHEMA_VERSION", "STUDY_ID", "VARIANTS"]
