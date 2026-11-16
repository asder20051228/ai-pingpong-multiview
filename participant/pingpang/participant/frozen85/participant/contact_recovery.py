"""Recover uncovered contacts without changing the primary tracker output."""
from __future__ import annotations

import math
from typing import Mapping

from .contact_tracker import ContactTracker


class ContactRecovery:
    def __init__(self, rule: dict, table_margin: float = 0.0) -> None:
        self.rule = dict(rule)
        self.table_margin = float(table_margin)
        self.points: dict[int, tuple[float, float, float]] = {}
        self.primary: list[dict] = []

    def reset(self) -> None:
        self.points.clear()
        self.primary.clear()

    def record(self, frame: int, point) -> None:
        x, y, confidence = (float(value) for value in point)
        if confidence >= self.rule.get("traj_conf", 0.5):
            self.points[int(frame)] = (x, y, confidence)

    def remember(self, hit: dict) -> None:
        self.primary.append(dict(hit))

    def _fill_short_gaps(self) -> tuple[dict, set[int]]:
        points = dict(self.points)
        observed = sorted(self.points)
        filled: set[int] = set()
        for left_frame, right_frame in zip(observed, observed[1:]):
            distance = right_frame - left_frame
            if not 1 < distance <= 3:
                continue
            left, right = points[left_frame], points[right_frame]
            if min(left[2], right[2]) < 0.75:
                continue
            if math.dist(left[:2], right[:2]) / distance > self.rule["max_step"]:
                continue
            before, after = points.get(left_frame - 1), points.get(right_frame + 1)
            if before is None or after is None or min(before[2], after[2]) < self.rule["conf_th"]:
                continue
            vx = (left[0] - before[0], (right[0] - left[0]) / distance, after[0] - right[0])
            if min(vx) < -2.0 and max(vx) > 2.0:
                continue
            for frame in range(left_frame + 1, right_frame):
                fraction = (frame - left_frame) / distance
                points[frame] = (left[0] * (1 - fraction) + right[0] * fraction,
                                 left[1] * (1 - fraction) + right[1] * fraction,
                                 min(left[2], right[2]))
                filled.add(frame)
        return points, filled

    def _jitter_impulse(self, points: dict, frame: int) -> dict | None:
        ps = [points.get(f) for f in range(frame - 2, frame + 3)]
        if any(p is None or p[2] < self.rule["vertex_conf"] for p in ps):
            return None
        dx = [b[0] - a[0] for a, b in zip(ps, ps[1:])]
        dy = [b[1] - a[1] for a, b in zip(ps, ps[1:])]
        strength = dy[1] - dy[2]
        if min(dx) < -5.0 and max(dx) > 5.0:
            return None
        if strength < self.rule["velocity_impulse"] or strength < max(dy[0] - dy[1], dy[2] - dy[3]):
            return None
        if abs(dx[2] - dx[1]) > strength:
            return None
        if max(math.hypot(x, y) for x, y in zip(dx, dy)) > self.rule["max_step"]:
            return None
        return {"frame_id": frame, "x": ps[2][0], "y": ps[2][1],
                "confidence": min(p[2] for p in ps), "fall": strength,
                "rise": strength, "strength": strength, "kind": "velocity_impulse"}

    def _inside_table(self, candidate: dict, boxes: Mapping) -> bool:
        box = boxes.get(candidate["frame_id"])
        margin = self.table_margin
        return box is None or (box[0] - margin <= candidate["x"] <= box[2] + margin
                               and box[1] - margin <= candidate["y"] <= box[3] + margin)

    def _conflicts(self, a: dict, b: dict) -> bool:
        return (abs(a["frame_id"] - b["frame_id"]) < self.rule["min_gap"]
                and math.hypot(a["x"] - b["x"], a["y"] - b["y"]) <= 100.0)

    def recover(self, table_boxes: Mapping) -> list[dict]:
        points, filled = self._fill_short_gaps()
        tracker = ContactTracker(**self.rule)
        tracker.points = points
        candidates: list[dict] = []
        for frame in sorted(points):
            candidate = tracker._score(frame)
            if candidate is None:
                candidate = self._jitter_impulse(points, frame)
            if candidate is None:
                continue
            before, after = points.get(frame - 1), points.get(frame + 1)
            if before is not None and after is not None:
                centre = points[frame]
                candidate["local_motion"] = min(math.dist(before[:2], centre[:2]), math.dist(centre[:2], after[:2]))
                dx_before, dx_after = centre[0] - before[0], after[0] - centre[0]
                candidate["horizontal_reversal"] = (min(before[2], centre[2], after[2]) >= self.rule["vertex_conf"]
                                                    and dx_before * dx_after < 0
                                                    and min(abs(dx_before), abs(dx_after)) > 2.0)
            uses_fill = any(f in filled for f in range(frame - 2, frame + 3))
            if candidate.get("kind") != "velocity_impulse" and not uses_fill:
                continue
            # Interpolation supports a decision but never supplies its output centre.
            if frame not in self.points:
                continue
            if not tracker._event_is_reliable(candidate) or not self._inside_table(candidate, table_boxes):
                continue
            candidates.append(candidate)
        accepted: list[dict] = []
        for candidate in sorted(candidates, key=tracker._rank, reverse=True):
            if any(self._conflicts(candidate, old) for old in self.primary + accepted):
                continue
            accepted.append(candidate)
        return accepted
