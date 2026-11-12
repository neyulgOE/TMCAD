#!/usr/bin/env python3
"""Overlay baseline C++ and hierarchy JSON projected CAD boxes."""

import json
import re
from pathlib import Path

import cv2


ROOT = Path("/home/huyen1905/TMCAD")
RGB = ROOT / "dump_rgb.png"
BASELINE = ROOT / "debug/neural_hough_baseline.log"
NEURAL_ONLY = ROOT / "debug/neural_no_hough.log"
HIERARCHY = ROOT / "debug/hierarchy_ab.json"
OUT = ROOT / "debug/pose_ab_overlay.png"


def parse_baseline(path: Path):
    pat = re.compile(r"corner (\d+): \(([-+0-9.eE]+), ([-+0-9.eE]+)\)")
    corners = [None] * 8
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            corners[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    if any(p is None for p in corners):
        raise RuntimeError("baseline corners are incomplete")
    return corners


def points(values):
    return [(float(p[0]), float(p[1])) for p in values]


def draw_box(image, corners, color, label, offset):
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in pairs:
        p0 = tuple(round(v) for v in corners[a])
        p1 = tuple(round(v) for v in corners[b])
        cv2.line(image, p0, p1, color, 2, cv2.LINE_AA)
    cv2.putText(image, label, (18, offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)


def main():
    image = cv2.imread(str(RGB))
    if image is None:
        raise RuntimeError("cannot load RGB dump")
    baseline = parse_baseline(BASELINE)
    neural_only = parse_baseline(NEURAL_ONLY)
    hierarchy = points(json.loads(HIERARCHY.read_text())["corners_2d"])
    draw_box(image, baseline, (0, 255, 0), "NEURAL + Hough anchor", 28)
    draw_box(image, neural_only, (255, 0, 0), "NEURAL + STL edge ICP (no Hough)", 52)
    draw_box(image, hierarchy, (255, 0, 255), "STL hierarchy + depth", 76)
    cv2.imwrite(str(OUT), image)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
