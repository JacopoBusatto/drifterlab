"""Candidate deployment clustering from exact observed trajectory starts."""

from .core import (
    candidate_clusters,
    infer_start_cohorts,
    nearest_neighbor_diagnostics,
    pairwise_start_metrics,
)

__all__ = [
    "candidate_clusters",
    "infer_start_cohorts",
    "nearest_neighbor_diagnostics",
    "pairwise_start_metrics",
]
