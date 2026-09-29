"""
zones.py — Dynamic Polygonal Zone Classification

Replaces the original static 4-quadrant heuristic with arbitrary user-defined
polygon zones.  Zones are persisted in a JSON file so they survive across runs
and can be edited through the UI or by hand.

Each zone is a dict:
    {"name": "Desk", "color": [0, 255, 0], "points": [[x1,y1], [x2,y2], ...]}
"""

import json
import os
import cv2
import numpy as np

# ── Default Zones (same quadrant layout as Task 4, but expressed as polygons) ─
_DEFAULT_ZONES_FOR_RESOLUTION = None  # lazily built per video resolution


def default_zones(img_width: int, img_height: int) -> list[dict]:
    """Four quadrant polygons sized to a video frame."""
    return _build_default_zones(img_width, img_height)


def hex_to_bgr(hex_color: str) -> list[int]:
    """Convert a ``#RRGGBB`` colour to an OpenCV BGR triple."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return [0, 255, 0]
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return [b, g, r]


def hex_to_rgba(hex_color: str, alpha: float = 0.35) -> str:
    """Convert ``#RRGGBB`` to a CSS rgba() string for the drawing canvas."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(0, 255, 0, {alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def canvas_polygon_points(
    obj: dict,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
) -> list[list[int]]:
    """Map a Fabric.js polygon from streamlit-drawable-canvas onto frame pixels.

    Canvas coordinates are display-sized. ``x_scale`` / ``y_scale`` convert
    them back to the original video frame (original / display).
    """
    raw = obj.get("points") or []
    if obj.get("type") not in (None, "polygon", "polyline", "path"):
        return []
    if len(raw) < 3:
        return []

    width = float(obj.get("width") or 0)
    height = float(obj.get("height") or 0)
    # Stray right-clicks on the canvas produce an empty polygon.
    if width < 2 or height < 2:
        return []

    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    sx = float(obj.get("scaleX") or 1)
    sy = float(obj.get("scaleY") or 1)
    offset = obj.get("pathOffset") or {}
    ox = float(offset.get("x", 0))
    oy = float(offset.get("y", 0))

    if obj.get("originX") == "center":
        cx = left
    else:
        cx = left + (width * sx) / 2.0
    if obj.get("originY") == "center":
        cy = top
    else:
        cy = top + (height * sy) / 2.0

    points: list[list[int]] = []
    for p in raw:
        x = cx + (float(p["x"]) - ox) * sx
        y = cy + (float(p["y"]) - oy) * sy
        points.append([int(round(x * x_scale)), int(round(y * y_scale))])
    return points


def _build_default_zones(img_width: int, img_height: int) -> list[dict]:
    """Return four quadrant-based polygon zones matching the Task 4 layout."""
    hw = img_width // 2
    hh = img_height // 2
    return [
        {
            "name": "Desk (Top-Left)",
            "color": [0, 255, 0],       # green
            "points": [[0, 0], [hw, 0], [hw, hh], [0, hh]],
        },
        {
            "name": "Bed (Top-Right)",
            "color": [255, 0, 0],       # blue (BGR)
            "points": [[hw, 0], [img_width, 0], [img_width, hh], [hw, hh]],
        },
        {
            "name": "Floor (Bottom-Left)",
            "color": [0, 165, 255],     # orange
            "points": [[0, hh], [hw, hh], [hw, img_height], [0, img_height]],
        },
        {
            "name": "Sofa (Bottom-Right)",
            "color": [255, 0, 255],     # magenta
            "points": [[hw, hh], [img_width, hh], [img_width, img_height], [hw, img_height]],
        },
    ]


class ZoneManager:
    """Manages a list of polygonal zones for spatial classification."""

    DEFAULT_CONFIG_PATH = "zones.json"

    def __init__(self, config_path: str | None = None,
                 img_width: int = 1920, img_height: int = 1080):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.img_width = img_width
        self.img_height = img_height
        self.zones: list[dict] = []
        self._load(img_width, img_height)

    # ── Persistence ──────────────────────────────────────────────────────────
    def _load(self, img_width: int, img_height: int):
        """Load zones from JSON config, or fall back to defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    self.zones = data
                    return
            except (json.JSONDecodeError, IOError):
                pass  # fall through to defaults
        # Generate defaults for the given resolution
        self.zones = _build_default_zones(img_width, img_height)

    def save(self, path: str | None = None):
        """Persist current zone configuration to disk."""
        target = path or self.config_path
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.zones, f, indent=2)

    # ── Classification ───────────────────────────────────────────────────────
    def classify_point(self, cx: float, cy: float) -> str:
        """Return the zone name for the given centre point, or a fallback."""
        for zone in self.zones:
            polygon = np.array(zone["points"], dtype=np.float32)
            # pointPolygonTest returns  +1 inside, 0 on edge, -1 outside
            result = cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False)
            if result >= 0:
                return zone["name"]
        return "Unknown / In Transit"

    # ── Visualisation helpers ────────────────────────────────────────────────
    def draw_zones(self, frame: np.ndarray, alpha: float = 0.20) -> np.ndarray:
        """Draw translucent polygon overlays + labels on *frame* (in-place)."""
        overlay = frame.copy()
        for zone in self.zones:
            pts = np.array(zone["points"], dtype=np.int32)
            color = tuple(zone["color"])
            cv2.fillPoly(overlay, [pts], color)
            # Label at centroid
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cx_label = int(M["m10"] / M["m00"])
                cy_label = int(M["m01"] / M["m00"])
            else:
                cx_label, cy_label = pts[0][0], pts[0][1]
            cv2.putText(overlay, zone["name"],
                        (cx_label - 40, cy_label),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # Draw polygon borders (solid, full opacity)
        for zone in self.zones:
            pts = np.array(zone["points"], dtype=np.int32)
            color = tuple(zone["color"])
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        return frame

    def get_zone_names(self) -> list[str]:
        """Return ordered list of zone names."""
        return [z["name"] for z in self.zones]

    def update_zones(self, new_zones: list[dict]):
        """Replace the current zone list and persist."""
        self.zones = new_zones
        self.save()

    def to_dict_list(self) -> list[dict]:
        """Serialisable representation."""
        return self.zones


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    zm = ZoneManager(img_width=1280, img_height=720)
    print("Zones:", zm.get_zone_names())
    print("Centre (320, 180):", zm.classify_point(320, 180))     # Desk
    print("Centre (960, 180):", zm.classify_point(960, 180))     # Bed
    print("Centre (320, 540):", zm.classify_point(320, 540))     # Floor
    print("Centre (960, 540):", zm.classify_point(960, 540))     # Sofa
    zm.save("zones_test.json")
    print("Saved test config to zones_test.json")
