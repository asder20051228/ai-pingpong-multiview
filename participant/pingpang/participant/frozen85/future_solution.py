"""Separate ablation: only future-context ch0 may rescue missing ch1 points."""
from .experiment_solution import Solution as MultiChannelSolution

class Solution(MultiChannelSolution):
    def finish_video(self,meta):
        self.candidates={f:[p for p in candidates if p['channel']==0]
                         for f,candidates in self.candidates.items()}
        return super().finish_video(meta)
