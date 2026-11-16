"""Ablation only: identical TensorRT models and tracker, without table filtering."""
from core.types import Landing
from .solution import Solution as GatedSolution


class Solution(GatedSolution):
    def _gate(self, hits):
        return [Landing(hit["frame_id"], hit["x"], hit["y"]) for hit in hits]
