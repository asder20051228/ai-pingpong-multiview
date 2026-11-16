"""Table-contact rule over a frame-indexed ball trajectory.

A contact is a local maximum of the image-space y coordinate, scored over a
symmetric window of +/- ``window`` frames so that shallow bounces survive
detector noise. Emitting is deferred: two vertices closer than ``min_gap``
frames compete and the stronger one wins, which is what an offline
non-maximum suppression pass would choose.

Optional velocity-impulse and short-window branches handle perspective motion
and missing outer observations; both are disabled for legacy callers.
"""
from __future__ import annotations

from typing import Iterable


class ContactTracker:
    def __init__(self, window: int = 2, conf_th: float = 0.75, min_gap: int = 7,
                 min_fall: float = 0.5, min_rise: float = 0.5,
                 max_jump: float = 110.0, traj_conf: float = 0.5,
                 vertex_conf: float | None = None,
                 rescue_vertex_conf: float | None = None,
                 velocity_impulse: float | None = None,
                 max_step: float | None = None,
                 short_window_fallback: bool = False,
                 secondary_impulse: bool = False,
                 min_contact_motion: float | None = None,
                 reject_contact_reversal: bool = False) -> None:
        self.window = int(window)
        self.conf_th = float(conf_th)
        # Supporting frames describe trajectory shape; the contact vertex must
        # satisfy its own confidence threshold before entering candidate NMS.
        # None preserves the original single-threshold behavior for callers.
        self.vertex_conf = float(conf_th if vertex_conf is None else vertex_conf)
        self.rescue_vertex_conf = (None if rescue_vertex_conf is None
                                   else float(rescue_vertex_conf))
        self.velocity_impulse = (None if velocity_impulse is None
                                 else float(velocity_impulse))
        self.max_step = None if max_step is None else float(max_step)
        self.short_window_fallback = bool(short_window_fallback)
        self.secondary_impulse = bool(secondary_impulse)
        self.min_contact_motion = (None if min_contact_motion is None
                                   else float(min_contact_motion))
        self.reject_contact_reversal = bool(reject_contact_reversal)
        self.min_gap = int(min_gap)
        self.min_fall = float(min_fall)
        self.min_rise = float(min_rise)
        self.max_jump = float(max_jump)
        self.traj_conf = float(traj_conf)
        self.points: dict[int, tuple[float, float, float]] = {}
        self.scored_through = -1
        self.pending: dict | None = None

    # ── input ───────────────────────────────────────────────────────────────
    def add(self, frame: int, point: Iterable[float] | None) -> None:
        """Record a ball centre for `frame`; None means 'not detected'."""
        if point is None:
            return
        x, y, conf = (float(value) for value in point)
        if conf < self.traj_conf:
            return
        self.points[int(frame)] = (x, y, conf)
        cutoff = self.scored_through + 1 - self.window
        for stale in [f for f in self.points if f < cutoff]:
            del self.points[stale]

    # ── scoring ─────────────────────────────────────────────────────────────
    @staticmethod
    def _strength(candidate: dict) -> float:
        return candidate.get("strength", min(candidate["fall"], candidate["rise"]))

    @classmethod
    def _rank(cls, candidate: dict) -> tuple[int, float]:
        # Lower-confidence fallback candidates never replace primary candidates.
        return (0 if candidate.get("kind") == "secondary_impulse" else 1,
                cls._strength(candidate))

    def _event_is_reliable(self, candidate: dict) -> bool:
        if self.min_contact_motion is not None and candidate.get("local_motion", float("inf")) < self.min_contact_motion:
            return False
        return not (self.reject_contact_reversal and candidate.get("horizontal_reversal", False))

    def _vertex_is_reliable(self, vertex: int, centre: tuple) -> bool:
        if centre[2] >= self.vertex_conf:
            return True
        if self.rescue_vertex_conf is None or centre[2] < self.rescue_vertex_conf:
            return False
        # A weaker vertex needs confident immediate neighbours that independently
        # confirm the bounce. The usual wider-window checks still apply below.
        left = self.points.get(vertex - 1)
        right = self.points.get(vertex + 1)
        return (left is not None and right is not None
                and min(left[2], right[2]) >= self.vertex_conf
                and centre[1] - left[1] >= self.min_fall
                and centre[1] - right[1] >= self.min_rise)

    def _score(self, vertex: int) -> dict | None:
        candidate = self._score_peak(vertex)
        if candidate is None and self.velocity_impulse is not None:
            candidate = self._score_impulse(vertex)
        if candidate is None and self.short_window_fallback:
            candidate = self._score_short_window(vertex)
        if candidate is None and self.secondary_impulse and self.velocity_impulse is not None:
            candidate = self._score_impulse(vertex, secondary=True)
        if candidate is not None and self.max_step is not None:
            centre = self.points[vertex]
            # The wider window can hide a one-frame jump to a different object.
            for frame in (vertex - 1, vertex + 1):
                neighbour = self.points.get(frame)
                if neighbour is not None and self._distance(centre, neighbour) > self.max_step:
                    return None
        if candidate is not None and (self.min_contact_motion is not None or self.reject_contact_reversal):
            before, centre, after = [self.points.get(f) for f in (vertex - 1, vertex, vertex + 1)]
            if before is not None and after is not None:
                # Keep evidence with the candidate: points may be pruned before
                # NMS emits it. Apply these quality checks after candidate choice.
                candidate["local_motion"] = min(self._distance(before, centre), self._distance(centre, after))
                dx_before, dx_after = centre[0] - before[0], after[0] - centre[0]
                candidate["horizontal_reversal"] = (min(before[2], centre[2], after[2]) >= self.vertex_conf
                                                    and dx_before * dx_after < 0
                                                    and min(abs(dx_before), abs(dx_after)) > 2.0)
        return candidate

    @staticmethod
    def _distance(a: tuple, b: tuple) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _score_impulse(self, vertex: int, secondary: bool = False) -> dict | None:
        # Perspective motion can keep y increasing after a table impact.
        # Detect the upward velocity change using five confident observations.
        if self.window < 2:
            return None
        points = [self.points.get(frame) for frame in range(vertex - 2, vertex + 3)]
        if any(p is None for p in points):
            return None
        if secondary:
            # Only one outer observation may be weaker; the centre and its
            # immediate neighbours still need the strong confidence threshold.
            if (min(points[0][2], points[4][2]) < self.conf_th
                    or max(points[0][2], points[4][2]) < self.vertex_conf
                    or min(p[2] for p in points[1:4]) < self.vertex_conf):
                return None
        elif any(p[2] < self.vertex_conf for p in points):
            return None
        dx = [b[0] - a[0] for a, b in zip(points, points[1:])]
        dy = [b[1] - a[1] for a, b in zip(points, points[1:])]
        if min(dx) < -2.0 and max(dx) > 2.0:
            return None
        impulse = dy[1] - dy[2]
        if impulse < self.velocity_impulse:
            return None
        if impulse < max(dy[0] - dy[1], dy[2] - dy[3]):
            return None
        if abs(dx[2] - dx[1]) > impulse:
            return None
        speed = max((x * x + y * y) ** 0.5 for x, y in zip(dx, dy))
        if speed > (self.max_step if self.max_step is not None else self.max_jump * 2):
            return None
        centre = points[2]
        return {"frame_id": vertex, "x": centre[0], "y": centre[1],
                "confidence": min(p[2] for p in points),
                "fall": centre[1] - points[0][1], "rise": centre[1] - points[4][1],
                "speed": speed, "strength": impulse,
                "kind": "secondary_impulse" if secondary else "velocity_impulse"}

    def _score_short_window(self, vertex: int) -> dict | None:
        if self.window < 2:
            return None
        if self.points.get(vertex - self.window) is not None and self.points.get(vertex + self.window) is not None:
            return None
        before, centre, after = [self.points.get(frame) for frame in (vertex - 1, vertex, vertex + 1)]
        if before is None or centre is None or after is None:
            return None
        if centre[2] < self.vertex_conf or min(before[2], after[2]) < self.conf_th:
            return None
        fall, rise = centre[1] - before[1], centre[1] - after[1]
        if fall < self.min_fall or rise < self.min_rise:
            return None
        dx_before, dx_after = centre[0] - before[0], after[0] - centre[0]
        if dx_before * dx_after < 0 and min(abs(dx_before), abs(dx_after)) > 2.0:
            return None
        speed = max(self._distance(before, centre), self._distance(centre, after))
        if speed > (self.max_step if self.max_step is not None else self.max_jump * 2):
            return None
        return {"frame_id": vertex, "x": centre[0], "y": centre[1],
                "confidence": min(before[2], centre[2], after[2]),
                "fall": fall, "rise": rise, "speed": speed, "kind": "short_window"}

    def _score_peak(self, vertex: int) -> dict | None:
        centre = self.points.get(vertex)
        before = self.points.get(vertex - self.window)
        after = self.points.get(vertex + self.window)
        if centre is None or before is None or after is None:
            return None
        if not self._vertex_is_reliable(vertex, centre):
            return None
        if min(centre[2], before[2], after[2]) < self.conf_th:
            return None
        fall = centre[1] - before[1]
        rise = centre[1] - after[1]
        if fall < self.min_fall or rise < self.min_rise:
            return None
        if centre[1] < before[1] or centre[1] < after[1]:
            return None
        speed_before = ((centre[0] - before[0]) ** 2 + fall ** 2) ** 0.5 / self.window
        speed_after = ((after[0] - centre[0]) ** 2 + rise ** 2) ** 0.5 / self.window
        if max(speed_before, speed_after) > self.max_jump:
            return None
        return {"frame_id": vertex, "x": centre[0], "y": centre[1],
                "confidence": min(centre[2], before[2], after[2]),
                "fall": fall, "rise": rise,
                "speed": max(speed_before, speed_after)}

    def advance(self, current_frame: int) -> list[dict]:
        """Score every vertex whose forward window is complete; return confirmed contacts."""
        confirmed: list[dict] = []
        last_vertex = int(current_frame) - self.window
        while self.scored_through < last_vertex:
            self.scored_through += 1
            candidate = self._score(self.scored_through)
            if candidate is None:
                continue
            if self.pending is None:
                self.pending = candidate
            elif candidate["frame_id"] - self.pending["frame_id"] < self.min_gap:
                if self._rank(candidate) > self._rank(self.pending):
                    self.pending = candidate
            else:
                if self._event_is_reliable(self.pending):
                    confirmed.append(self.pending)
                self.pending = candidate
        return confirmed

    def flush(self) -> list[dict]:
        """Release the buffered candidate at the end of a video."""
        confirmed = []
        if self.pending is not None:
            if self._event_is_reliable(self.pending):
                confirmed.append(self.pending)
            self.pending = None
        return confirmed

    def reset(self) -> None:
        self.points.clear()
        self.scored_through = -1
        self.pending = None
