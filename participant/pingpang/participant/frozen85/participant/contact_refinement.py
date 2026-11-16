"""Subframe contact localization and conservative trajectory consistency checks."""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np


class ContactRefinement:
    def __init__(self, rule: dict, table_margin: float = 0.0) -> None:
        self.rule = dict(rule)
        self.table_margin = float(table_margin)
        self._times = np.arange(-3, 4, dtype=float)
        self._offsets = np.linspace(-0.9, 0.9, 73)
        self._designs = [np.column_stack((np.ones(7), self._times, self._times**2,
                                         np.maximum(self._times-offset, 0)))
                         for offset in self._offsets]

    def _inside_table(self, candidate: dict, boxes: Mapping) -> bool:
        box = boxes.get(candidate['frame_id'])
        margin = self.table_margin
        return box is None or (box[0]-margin <= candidate['x'] <= box[2]+margin
                               and box[1]-margin <= candidate['y'] <= box[3]+margin)

    def _conflicts(self, a: dict, b: dict) -> bool:
        return (abs(a['frame_id']-b['frame_id']) < self.rule['min_gap']
                and math.hypot(a['x']-b['x'], a['y']-b['y']) <= 100.0)

    @staticmethod
    def _horizontal_impact(points: Mapping, hit: dict) -> bool:
        frame = hit['frame_id']
        centre = points.get(frame)
        if centre is None or centre[2] < 0.8:
            return False
        left = next((f for f in range(frame-1, frame-3, -1) if f in points), None)
        right = next((f for f in range(frame+1, frame+3) if f in points), None)
        if left is None or right is None or min(points[left][2], points[right][2]) < 0.8:
            return False
        vx0 = (centre[0]-points[left][0])/(frame-left)
        vx1 = (points[right][0]-centre[0])/(right-frame)
        vy0 = (centre[1]-points[left][1])/(frame-left)
        vy1 = (points[right][1]-centre[1])/(right-frame)
        # A horizontal reversal alone also occurs in real table contacts.
        # Reject only when the horizontal impulse dominates the vertical one.
        return vx0*vx1 < 0 and min(abs(vx0), abs(vx1)) > 2 and abs(vx1-vx0) > abs(vy1-vy0)

    def _fit_candidate(self, points: Mapping, frame: int, min_speed: float) -> dict | None:
        ps = [points.get(f) for f in range(frame-3, frame+4)]
        if any(p is None or p[2] < 0.75 for p in ps) or sum(p[2] >= 0.8 for p in ps) < 6:
            return None
        dx = [b[0]-a[0] for a, b in zip(ps, ps[1:])]
        if min(dx) < -2 and max(dx) > 2:
            return None
        if abs(ps[4][0]-ps[2][0])/2 < min_speed:
            return None
        if max(math.dist(a[:2], b[:2]) for a, b in zip(ps, ps[1:])) > self.rule['max_step']:
            return None
        ys = np.array([p[1] for p in ps])
        best = None
        # Shared acceleration with a vertical velocity change at continuous time tau:
        # y(t) = a + b*t + c*t*t + d*max(t-tau, 0).
        for offset, design in zip(self._offsets, self._designs):
            coefficients = np.linalg.lstsq(design, ys, rcond=None)[0]
            error = float(np.mean((design @ coefficients-ys)**2))
            if best is None or error < best['error']:
                best = dict(offset=float(offset), error=error, impulse=float(-coefficients[3]),
                            gravity=float(2*coefficients[2]),
                            x=float(np.interp(offset, self._times, [p[0] for p in ps])),
                            y=float(coefficients[0]+coefficients[1]*offset+coefficients[2]*offset**2))
        if (best['error'] > 4 or not 0 <= best['gravity'] <= 20
                or best['impulse'] < 10 or abs(best['offset']) >= 0.875):
            return None
        return dict(frame_id=int(math.floor(frame+best['offset']+0.5)), x=best['x'], y=best['y'],
                    fit=best, source_frame=frame)

    def refine(self, hits: list[dict], points: Mapping, table_boxes: Mapping) -> list[dict]:
        output: list[dict] = []
        for hit in hits:
            if self._horizontal_impact(points, hit):
                continue
            candidate = self._fit_candidate(points, hit['frame_id'], min_speed=80.0)
            if (candidate is not None and abs(candidate['fit']['offset']) >= 0.35
                    and self._inside_table(candidate, table_boxes)
                    and not any(self._conflicts(candidate, other) for other in hits if other is not hit)):
                output.append(candidate)
            else:
                output.append(dict(hit))
        candidates = [candidate for frame in sorted(points)
                      if (candidate := self._fit_candidate(points, frame, min_speed=40.0)) is not None
                      and self._inside_table(candidate, table_boxes)]
        for candidate in sorted(candidates, key=lambda c: c['fit']['impulse'], reverse=True):
            # Keep original anchors even when a prediction is filtered/refined:
            # changing that competition could expose an old suppressed false hit.
            if any(self._conflicts(candidate, old) for old in hits+output):
                continue
            output.append(candidate)
        return output
