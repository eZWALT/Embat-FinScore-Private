"""Feature families. Each family module exposes build(con, grid) -> DataFrame.

Contract: output has keys company_id, period plus family columns only.
No look-ahead past period end. Prefix columns with family letter: a_, b_, ...
"""
from .common import connect, load_holdout, train_mask
from .grid import company_meta, monthly_grid, weekly_grid

__all__ = [
    "connect",
    "load_holdout",
    "train_mask",
    "monthly_grid",
    "weekly_grid",
    "company_meta",
]
