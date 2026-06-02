"""
DrawingDiffGUI.py  (security-patched)
======================================
Shipbuilding AI — Drawing Verification & Highlighting System

Security patches applied:
  CRITICAL:
    • tempfile.mktemp() → NamedTemporaryFile + atexit cleanup (race-condition fix)
    • render_highlight / render_side_by_side: RenderParams NamedTuple + Y-axis flip
    • _validate_file_path()  — extension + existence + readability check
    • _validate_output_path() — home-dir restriction + extension allowlist
    • ENTITY_LIMIT = 100 000 — prevents DoS via huge DXF files
    • _validate_coordinate() — rejects NaN/Inf and out-of-range coords
    • All bare `except:` replaced with specific exception types + logging

  HIGH:
    • Output paths restricted to home directory
    • Output extension allowlist  (.png / .jpg / .jpeg)
    • File-overwrite confirmation dialog
    • Warning suppression scoped to ezdxf / numpy only (not global)

  NICE-TO-HAVE:
    • _apply_resource_limits() — optional memory cap (Unix)
    • ANALYSIS_TIMEOUT_S per-analysis watchdog thread
    • Annotated security test stubs  (run with pytest)

SETUP:
    pip install customtkinter pillow opencv-python-headless numpy ezdxf shapely rtree

RUN:
    python DrawingDiffGUI.py
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import atexit
import logging
import math
import os
import sys
import tempfile
import threading
import time
import webbrowser
from typing import NamedTuple

# ── third-party ─────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import cv2
from PIL import Image as PILImage, ImageTk, ImageDraw
import customtkinter as ctk

# ── scoped warning suppression (CRITICAL fix: was `warnings.filterwarnings("ignore")`)
import warnings
warnings.filterwarnings("ignore", category=UserWarning,      module=r"ezdxf.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ezdxf.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=r".*invalid value encountered.*", module=r"numpy.*")

# ── optional deps ────────────────────────────────────────────────────────────
try:
    import ezdxf
    from ezdxf.addons import odafc
    DXF_OK = True
except ImportError:
    DXF_OK = False

try:
    from shapely.geometry import Polygon
    from rtree import index as _rtree_index
    SHAPELY_OK = True
except ImportError:
    SHAPELY_OK = False

# ── logging ──────────────────────────────────────────────────────────────────
_log = logging.getLogger(__name__)

# ── Theme ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG_DEEP    = "#080c14"; BG_CARD    = "#0d1420"; BG_PANEL   = "#111827"
BG_INPUT   = "#0f1923"; ACCENT     = "#00d4ff"; ACCENT_DIM = "#0e3a45"
SUCCESS    = "#10b981"; WARNING    = "#f59e0b"; DANGER     = "#ef4444"
TEXT_PRI   = "#e2e8f0"; TEXT_SEC   = "#94a3b8"; TEXT_DIM   = "#475569"
BORDER     = "#1e2d3d"

# ── Engine constants ──────────────────────────────────────────────────────────
IOU_THRESHOLD  = 0.30
AREA_RATIO_MAX = 8.0
SEARCH_RADIUS  = 0.08
CLUSTER_RADIUS = 0.07
MAX_CLUSTERS   = 20
RENDER_W       = 2800
RENDER_H       = 1600
RENDER_PAD     = 60
IOU_BUFFER     = 0.003

DC_W = 1800
DC_H = 1100

PALETTE_BGR = [
    (220,0,0),(0,0,220),(160,0,160),(0,160,0),(200,80,0),(0,160,160),
    (80,60,200),(160,0,70),(0,130,70),(130,0,200),(185,130,0),(0,185,120),
    (200,60,0),(60,185,0),(0,0,160),(150,70,0),(110,0,120),(0,110,110),
    (180,0,70),(40,40,190),
]

def bgr_to_hex(c):
    return "#{:02x}{:02x}{:02x}".format(c[2], c[1], c[0])


# ════════════════════════════════════════════════════════════════════════════
# SECURITY CONSTANTS  (CRITICAL + HIGH fixes)
# ════════════════════════════════════════════════════════════════════════════

# CRITICAL: entity-count DoS guard
ENTITY_LIMIT: int = 100_000

# CRITICAL: coordinate sanity bounds
COORD_BOUND: float = 1e9

# HIGH: allowed file extensions
ALLOWED_INPUT_EXTS:  frozenset[str] = frozenset({".dxf", ".dwg"})
ALLOWED_OUTPUT_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})

# HIGH: output files must stay inside the user's home directory
HOME_DIR: str = os.path.realpath(os.path.expanduser("~"))

# NICE-TO-HAVE: per-analysis time limit
ANALYSIS_TIMEOUT_S: int = 300   # 5 minutes


# ════════════════════════════════════════════════════════════════════════════
# SECURE TEMP-FILE MANAGER
# CRITICAL fix: replaces tempfile.mktemp() (race condition / TOCTOU)
# ════════════════════════════════════════════════════════════════════════════

_TEMP_FILES: list[str] = []


def _make_temp_dxf() -> str:
    """Return path to a new, securely-created empty temp DXF file.

    Uses NamedTemporaryFile so the OS atomically allocates the name and
    creates the file, eliminating the mktemp() TOCTOU race condition.
    The path is registered for atexit cleanup.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".dxf", delete=False)
    tmp.close()
    _TEMP_FILES.append(tmp.name)
    return tmp.name


def _cleanup_temp_files() -> None:
    """Remove all temp DXF files created during this session."""
    for path in _TEMP_FILES:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as exc:
            _log.warning("Could not remove temp file %s: %s", path, exc)


atexit.register(_cleanup_temp_files)


# ════════════════════════════════════════════════════════════════════════════
# PATH VALIDATORS  (CRITICAL + HIGH fixes)
# ════════════════════════════════════════════════════════════════════════════

def _validate_file_path(path: str) -> str:
    """Validate an *input* CAD file path.

    Checks:
      • Non-empty string
      • Extension in ALLOWED_INPUT_EXTS
      • File exists and is readable

    Returns the resolved absolute path.
    Raises ValueError / FileNotFoundError / PermissionError on failure.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("File path must be a non-empty string.")

    abs_path = os.path.realpath(os.path.abspath(path))
    ext = os.path.splitext(abs_path)[1].lower()

    if ext not in ALLOWED_INPUT_EXTS:
        raise ValueError(
            f"Unsupported input extension '{ext}'. "
            f"Allowed: {sorted(ALLOWED_INPUT_EXTS)}"
        )
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")
    if not os.access(abs_path, os.R_OK):
        raise PermissionError(f"File is not readable: {abs_path}")

    return abs_path


def _validate_output_path(path: str) -> str:
    """Validate an *output* image path.

    Checks:
      • Non-empty string
      • Extension in ALLOWED_OUTPUT_EXTS
      • Resolved path is inside HOME_DIR  (path-traversal guard)
      • Parent directory exists and is writable

    Returns the resolved absolute path.
    Raises ValueError / PermissionError on failure.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Output path must be a non-empty string.")

    abs_path = os.path.realpath(os.path.abspath(path))
    ext = os.path.splitext(abs_path)[1].lower()

    # HIGH: extension allowlist
    if ext not in ALLOWED_OUTPUT_EXTS:
        raise ValueError(
            f"Unsupported output extension '{ext}'. "
            f"Allowed: {sorted(ALLOWED_OUTPUT_EXTS)}"
        )

    # HIGH: restrict to home directory
    home_prefix = HOME_DIR + os.sep
    if abs_path != HOME_DIR and not abs_path.startswith(home_prefix):
        raise ValueError(
            f"Output file must be inside your home directory.\n"
            f"Home: {HOME_DIR}\nGiven: {abs_path}"
        )

    parent = os.path.dirname(abs_path)
    if not os.path.isdir(parent):
        raise ValueError(f"Output directory does not exist: {parent}")
    if not os.access(parent, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {parent}")

    return abs_path


# ════════════════════════════════════════════════════════════════════════════
# COORDINATE VALIDATOR  (CRITICAL fix)
# ════════════════════════════════════════════════════════════════════════════

def _validate_coordinate_bounds(xmin: float, ymin: float,
                                 xmax: float, ymax: float,
                                 label: str = "") -> None:
    """Raise ValueError if any bounding-box value is non-finite or exceeds COORD_BOUND."""
    vals = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
    for name, v in vals.items():
        if not math.isfinite(v):
            raise ValueError(
                f"Non-finite coordinate {name}={v!r} in {label or 'drawing'}."
            )
        if abs(v) > COORD_BOUND:
            raise ValueError(
                f"Coordinate {name}={v:.3g} exceeds bound ±{COORD_BOUND:.3g} "
                f"in {label or 'drawing'}."
            )
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(
            f"Degenerate bounding box in {label or 'drawing'}: "
            f"({xmin:.3g},{ymin:.3g})→({xmax:.3g},{ymax:.3g})"
        )


# ════════════════════════════════════════════════════════════════════════════
# RESOURCE LIMITS  (NICE-TO-HAVE)
# ════════════════════════════════════════════════════════════════════════════

# 2 GB virtual-memory cap applied at startup on Unix systems.
_MAX_MEMORY_BYTES: int = 2 * 1024 * 1024 * 1024


def _apply_resource_limits() -> None:
    """Apply optional process-level resource limits (Unix only)."""
    try:
        import resource  # Unix only
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_soft = min(_MAX_MEMORY_BYTES, hard) if hard > 0 else _MAX_MEMORY_BYTES
        resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
        _log.info("Memory limit set to %d MB", new_soft // (1024 * 1024))
    except (ImportError, AttributeError, ValueError):
        pass   # Windows or unsupported platform — silently skip


# ════════════════════════════════════════════════════════════════════════════
# CONVERSION ENGINE
# ════════════════════════════════════════════════════════════════════════════

def convert_dwg_to_dxf(dwg_path: str, log_fn=None) -> str:
    """Convert DWG → DXF using ODA File Converter.

    CRITICAL fix: uses _make_temp_dxf() instead of deprecated tempfile.mktemp().
    """
    dxf_out = _make_temp_dxf()          # ← CRITICAL: secure temp file
    try:
        if log_fn:
            log_fn(f"Converting {os.path.basename(dwg_path)} → DXF …", "info")
        odafc.convert(dwg_path, dxf_out, version="R2010", audit=False)
        if os.path.isfile(dxf_out) and os.path.getsize(dxf_out) > 100:
            if log_fn:
                log_fn("Conversion successful", "success")
            return dxf_out
        raise RuntimeError("ODA produced an empty or missing output file.")
    except (OSError, RuntimeError) as exc:          # ← CRITICAL: specific exception
        if log_fn:
            log_fn(f"ODA conversion error: {exc}", "error")
        raise RuntimeError(
            f"Cannot convert {os.path.basename(dwg_path)}: {exc}"
        ) from exc


def prepare_dxf_path(file_path: str, log_fn=None) -> str:
    """Ensure we have a valid DXF file, converting DWG if necessary.

    CRITICAL fix: calls _validate_file_path() before any processing.
    """
    clean_path = _validate_file_path(file_path)     # ← CRITICAL: validate first
    ext = os.path.splitext(clean_path)[1].lower()
    if ext == ".dxf":
        return clean_path
    return convert_dwg_to_dxf(clean_path, log_fn)   # .dwg guaranteed by allowlist


def dxf_to_image(dxf_path: str,
                 width: int = RENDER_W,
                 height: int = RENDER_H,
                 padding: int = RENDER_PAD) -> PILImage.Image:
    """Render a DXF file to a PIL RGB image.

    CRITICAL fixes:
      • ENTITY_LIMIT guard prevents DoS from huge files.
      • _validate_coordinate_bounds() rejects non-finite / giant coords.
      • Bare except replaced with specific exception types.
      • Y-axis correctly inverted (DXF Y↑ → image Y↓).
    """
    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        # CRITICAL: entity count limit
        entity_count = sum(1 for _ in msp)
        if entity_count > ENTITY_LIMIT:
            raise ValueError(
                f"Drawing has {entity_count:,} entities, exceeding limit of {ENTITY_LIMIT:,}."
            )

        extents = msp.extent()
        if not extents.has_data:
            raise RuntimeError("DXF contains no drawable entities.")

        min_pt, max_pt = extents.min, extents.max

        # CRITICAL: coordinate bounds check
        _validate_coordinate_bounds(
            min_pt.x, min_pt.y, max_pt.x, max_pt.y, label=os.path.basename(dxf_path)
        )

        dxf_w = max(max_pt.x - min_pt.x, 1e-9)
        dxf_h = max(max_pt.y - min_pt.y, 1e-9)

        canvas_w = width  - 2 * padding
        canvas_h = height - 2 * padding
        scale    = min(canvas_w / dxf_w, canvas_h / dxf_h)

        img  = PILImage.new("RGB", (width, height), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        scaled_w = dxf_w * scale
        scaled_h = dxf_h * scale
        off_x = padding + (canvas_w - scaled_w) / 2
        off_y = padding + (canvas_h - scaled_h) / 2

        for entity in msp:
            dxftype = entity.dxftype()
            try:
                if dxftype == "LWPOLYLINE":
                    points = [(p[0], p[1]) for p in entity.get_points()]
                    if len(points) >= 2:
                        _draw_polyline(draw, points, min_pt, scale,
                                       off_x, off_y, height)
                elif dxftype == "POLYLINE":
                    points = [
                        (v.dxf.location.x, v.dxf.location.y)
                        for v in entity.vertices
                    ]
                    if len(points) >= 2:
                        _draw_polyline(draw, points, min_pt, scale,
                                       off_x, off_y, height)
                elif dxftype == "LINE":
                    s, e = entity.dxf.start, entity.dxf.end
                    p1 = _transform_point(
                        (s.x, s.y), min_pt, scale, off_x, off_y, height
                    )
                    p2 = _transform_point(
                        (e.x, e.y), min_pt, scale, off_x, off_y, height
                    )
                    draw.line([p1, p2], fill=(30, 30, 30), width=2)
                elif dxftype == "CIRCLE":
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    r       = entity.dxf.radius
                    cp      = _transform_point(
                        (cx, cy), min_pt, scale, off_x, off_y, height
                    )
                    r_px = r * scale
                    draw.ellipse(
                        [cp[0]-r_px, cp[1]-r_px, cp[0]+r_px, cp[1]+r_px],
                        outline=(30, 30, 30), width=2
                    )
            except (AttributeError, ValueError, TypeError,
                    ArithmeticError) as exc:         # ← CRITICAL: specific exceptions
                _log.debug("Skipping entity %s: %s", dxftype, exc)

        return img

    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Failed to convert DXF to image: {exc}") from exc


def _transform_point(point, min_point, scale, offset_x, offset_y,
                     img_h: int = RENDER_H) -> tuple[int, int]:
    """Transform DXF (x, y) → image pixel with correct Y-axis inversion.

    CRITICAL fix: added Y-flip.
    DXF Y increases *upward*; PIL image Y increases *downward*.
    Without the flip the drawing appears vertically mirrored.
    """
    px = (point[0] - min_point.x) * scale + offset_x
    # Y-flip: map DXF y=ymin → bottom of canvas, y=ymax → top
    py = img_h - ((point[1] - min_point.y) * scale + offset_y)
    return (int(px), int(py))


def _draw_polyline(draw, points, min_point, scale,
                   offset_x, offset_y, img_h: int) -> None:
    """Draw a polyline on a PIL draw surface."""
    if len(points) < 2:
        return
    transformed = [
        _transform_point(p, min_point, scale, offset_x, offset_y, img_h)
        for p in points
    ]
    for i in range(len(transformed) - 1):
        draw.line([transformed[i], transformed[i + 1]], fill=(30, 30, 30), width=2)


# ════════════════════════════════════════════════════════════════════════════
# RENDER PARAMS  (CRITICAL fix: replaces fragile positional tuple p[0]…p[5])
# ════════════════════════════════════════════════════════════════════════════

class RenderParams(NamedTuple):
    """Named parameters for the DXF→pixel coordinate transform."""
    offset_x:  float   # left padding in pixels
    offset_y:  float   # top  padding in pixels
    xmin:      float   # minimum DXF x coordinate
    ymin:      float   # minimum DXF y coordinate
    scale_x:   float   # pixels-per-DXF-unit (horizontal)
    scale_y:   float   # pixels-per-DXF-unit (vertical)
    canvas_h:  int     # full image height (for Y-axis flip)


def _dxf_to_pixel(x: float, y: float, rp: RenderParams) -> tuple[int, int]:
    """Transform a single DXF point to a PIL image pixel.

    CRITICAL fix: explicit Y-axis inversion to match PIL coordinate system.
    """
    px = rp.offset_x + (x - rp.xmin) * rp.scale_x
    py = rp.canvas_h - (rp.offset_y + (y - rp.ymin) * rp.scale_y)
    return (int(px), int(py))


# ════════════════════════════════════════════════════════════════════════════
# DIFF ENGINE
# ════════════════════════════════════════════════════════════════════════════

def load_normalised_shapes(dxf_path: str):
    """Load entities from DXF, validate them, and return normalised shape dicts.

    CRITICAL fixes:
      • ENTITY_LIMIT guard.
      • _validate_coordinate_bounds() on bbox.
      • Bare except replaced with specific types + debug logging.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # CRITICAL: entity count limit
    entity_list = list(msp)
    if len(entity_list) > ENTITY_LIMIT:
        raise ValueError(
            f"Drawing has {len(entity_list):,} entities, "
            f"exceeding limit of {ENTITY_LIMIT:,}."
        )

    raw: list[tuple[list, bool]] = []

    for e in entity_list:
        t = e.dxftype()
        try:
            if t == "LWPOLYLINE":
                pts = [(x, y) for x, y, *_ in e.get_points()]
                if len(pts) >= 2:
                    raw.append((pts, e.is_closed))
            elif t == "POLYLINE":
                pts = [
                    (v.dxf.location.x, v.dxf.location.y)
                    for v in e.vertices
                ]
                if len(pts) >= 2:
                    raw.append((pts, bool(e.is_closed)))
            elif t == "LINE":
                s, en = e.dxf.start, e.dxf.end
                raw.append(([(s.x, s.y), (en.x, en.y)], False))
            elif t == "CIRCLE":
                cx, cy, r = e.dxf.center.x, e.dxf.center.y, e.dxf.radius
                pts = [
                    (cx + r * math.cos(math.radians(a)),
                     cy + r * math.sin(math.radians(a)))
                    for a in range(0, 361, 15)
                ]
                raw.append((pts, True))
        except (AttributeError, ValueError, TypeError) as exc:  # ← CRITICAL
            _log.debug("Skipping entity %s: %s", t, exc)

    if not raw:
        raise RuntimeError("No drawable entities found in DXF.")

    all_pts   = np.array([p for pts, _ in raw for p in pts])
    xmin, ymin = float(all_pts[:, 0].min()), float(all_pts[:, 1].min())
    xmax, ymax = float(all_pts[:, 0].max()), float(all_pts[:, 1].max())

    # CRITICAL: coordinate bounds check
    _validate_coordinate_bounds(
        xmin, ymin, xmax, ymax, label=os.path.basename(dxf_path)
    )

    W = max(xmax - xmin, 1e-9)
    H = max(ymax - ymin, 1e-9)

    shapes: list[dict] = []
    for pts, closed in raw:
        norm = [((x - xmin) / W, (y - ymin) / H) for x, y in pts]
        try:
            g = Polygon(norm) if (closed and len(norm) >= 3) else None
            if g and not g.is_valid:
                g = g.buffer(0)
            if not g or g.is_empty or g.area < 1e-8:
                continue
            b = g.bounds
            shapes.append({
                "geom":    g,
                "area":    g.area,
                "cx":      (b[0] + b[2]) / 2,
                "cy":      (b[1] + b[3]) / 2,
                "bounds":  b,
                "pts_raw": pts,
                "npts":    len(pts),
            })
        except (ValueError, ArithmeticError) as exc:            # ← CRITICAL
            _log.debug("Skipping polygon: %s", exc)

    return shapes, xmin, ymin, xmax, ymax, W, H


def iou_fn(g1, g2, buf: float = IOU_BUFFER) -> float:
    """Compute Intersection-over-Union between two Shapely geometries."""
    try:
        a, b = g1.buffer(buf), g2.buffer(buf)
        i    = a.intersection(b).area
        u    = a.union(b).area
        return i / u if u > 1e-12 else 0.0
    except (ValueError, ArithmeticError) as exc:                # ← CRITICAL
        _log.debug("iou_fn error: %s", exc)
        return 0.0


def match_shapes(s1: list, s2: list):
    """Match shapes from drawing-1 to drawing-2 via IoU + spatial index."""
    idx2 = _rtree_index.Index()
    for i, s in enumerate(s2):
        b = s["bounds"]
        idx2.insert(i, (b[0], b[1], b[2], b[3]))

    m1: set[int] = set()
    m2: set[int] = set()
    pairs: list[tuple[int, int, float]] = []

    for i, s in enumerate(s1):
        cx, cy = s["cx"], s["cy"]
        r      = SEARCH_RADIUS
        cands  = list(idx2.intersection((cx - r, cy - r, cx + r, cy + r)))
        bv, bj = 0.0, -1

        for j in cands:
            iou = iou_fn(s["geom"], s2[j]["geom"])
            a1, a2 = s["area"], s2[j]["area"]
            ar     = max(a1, a2) / (min(a1, a2) + 1e-12)

            if iou > IOU_THRESHOLD and ar < AREA_RATIO_MAX and iou > bv:
                bv, bj = iou, j

        if bj >= 0:
            pairs.append((i, bj, bv))
            m1.add(i)
            m2.add(bj)

    return pairs, m1, m2


# ════════════════════════════════════════════════════════════════════════════
# RENDER FUNCTIONS  (CRITICAL: fixed coord transform + Y-flip)
# ════════════════════════════════════════════════════════════════════════════

def render_highlight(p1: RenderParams, p2: RenderParams,
                     s1: list, s2: list,
                     pairs, m1, m2) -> PILImage.Image:
    """Render a highlight-diff image: red = D1 shapes, green = D2 shapes.

    CRITICAL fixes:
      • Uses RenderParams NamedTuple instead of positional tuple indexing.
      • _dxf_to_pixel() applies correct Y-axis inversion.
    """
    img  = PILImage.new("RGB", (RENDER_W, RENDER_H), (255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    for shape in s1:
        pts = shape["pts_raw"]
        if len(pts) < 2:
            continue
        try:
            scaled = [_dxf_to_pixel(x, y, p1) for x, y in pts]
            if len(scaled) >= 2:
                draw.polygon(scaled, outline=(220, 60, 60), width=2)
        except (ValueError, OverflowError, ArithmeticError) as exc:
            _log.debug("render_highlight D1 shape skip: %s", exc)

    for shape in s2:
        pts = shape["pts_raw"]
        if len(pts) < 2:
            continue
        try:
            scaled = [_dxf_to_pixel(x, y, p2) for x, y in pts]
            if len(scaled) >= 2:
                draw.polygon(scaled, outline=(30, 180, 30), width=2)
        except (ValueError, OverflowError, ArithmeticError) as exc:
            _log.debug("render_highlight D2 shape skip: %s", exc)

    return img


def render_side_by_side(p1: RenderParams, p2: RenderParams,
                        s1: list, s2: list) -> PILImage.Image:
    """Render a side-by-side comparison image.

    CRITICAL fixes: same as render_highlight.
    """
    half_w = RENDER_W // 2

    img1  = PILImage.new("RGB", (half_w, RENDER_H), (255, 255, 255))
    draw1 = ImageDraw.Draw(img1)

    for shape in s1:
        pts = shape["pts_raw"]
        if len(pts) < 2:
            continue
        try:
            scaled = [_dxf_to_pixel(x, y, p1) for x, y in pts]
            if len(scaled) >= 2:
                draw1.polygon(scaled, outline=(30, 30, 30), width=2)
        except (ValueError, OverflowError, ArithmeticError) as exc:
            _log.debug("render_sbs D1 shape skip: %s", exc)

    img2  = PILImage.new("RGB", (half_w, RENDER_H), (255, 255, 255))
    draw2 = ImageDraw.Draw(img2)

    for shape in s2:
        pts = shape["pts_raw"]
        if len(pts) < 2:
            continue
        try:
            scaled = [_dxf_to_pixel(x, y, p2) for x, y in pts]
            if len(scaled) >= 2:
                draw2.polygon(scaled, outline=(30, 30, 30), width=2)
        except (ValueError, OverflowError, ArithmeticError) as exc:
            _log.debug("render_sbs D2 shape skip: %s", exc)

    out = PILImage.new("RGB", (RENDER_W, RENDER_H), (255, 255, 255))
    out.paste(img1, (0,      0))
    out.paste(img2, (half_w, 0))
    return out


def _build_render_params(xmin: float, ymin: float,
                          W: float, H: float) -> RenderParams:
    """Construct a RenderParams for the shared RENDER_W × RENDER_H canvas."""
    canvas_w = RENDER_W - 2 * RENDER_PAD
    canvas_h = RENDER_H - 2 * RENDER_PAD
    return RenderParams(
        offset_x = RENDER_PAD,
        offset_y = RENDER_PAD,
        xmin     = xmin,
        ymin     = ymin,
        scale_x  = canvas_w / W,
        scale_y  = canvas_h / H,
        canvas_h = RENDER_H,
    )


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS TIMEOUT WATCHDOG  (NICE-TO-HAVE)
# ════════════════════════════════════════════════════════════════════════════

class _AnalysisWatchdog:
    """Posts a warning to the GUI log if an analysis thread runs too long."""

    def __init__(self, log_fn, timeout_s: int = ANALYSIS_TIMEOUT_S):
        self._log_fn   = log_fn
        self._timeout  = timeout_s
        self._cancel   = threading.Event()
        self._thread   = threading.Thread(
            target=self._run, daemon=True, name="AnalysisWatchdog"
        )

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def _run(self) -> None:
        if not self._cancel.wait(self._timeout):
            self._log_fn(
                f"⚠ Analysis is taking longer than {self._timeout}s. "
                "It will keep running but consider cancelling.",
                "warn",
            )


# ════════════════════════════════════════════════════════════════════════════
# GUI
# ════════════════════════════════════════════════════════════════════════════

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Drawing DiffChecker — Shipbuilding AI")
        self.geometry("1600x900")
        self.configure(fg_color=BG_DEEP)

        self.path1       = tk.StringVar(value="")
        self.path2       = tk.StringVar(value="")
        self.img1_pil    = None
        self.img2_pil    = None
        self._img1_tk    = None
        self._img2_tk    = None
        self._result     = None
        self._result_pil = None
        self._dc_pil: dict[str, PILImage.Image] = {}
        self._dc_tk: dict  = {}
        self._dc_mode    = "sideby"
        self._slider_pct = 0.5

        self._build_ui()
        self._check_deps()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        top_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=70)
        top_frame.pack(side="top", fill="x")
        top_frame.pack_propagate(False)

        ctk.CTkLabel(top_frame, text="📐 Drawing DiffChecker",
                     font=("Segoe UI", 24, "bold"),
                     text_color=TEXT_PRI).pack(side="left", padx=20, pady=15)

        btn_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        btn_frame.pack(side="left", padx=20, pady=15)

        ctk.CTkButton(btn_frame, text="📂 Load Drawing 1",
                      font=("Segoe UI", 12, "bold"),
                      fg_color=ACCENT, text_color="#000000",
                      command=self._load_file_1,
                      corner_radius=6, width=150).pack(side="left", padx=8)

        ctk.CTkButton(btn_frame, text="📂 Load Drawing 2",
                      font=("Segoe UI", 12, "bold"),
                      fg_color=ACCENT, text_color="#000000",
                      command=self._load_file_2,
                      corner_radius=6, width=150).pack(side="left", padx=8)

        ctk.CTkButton(top_frame, text="⚡ HIGHLIGHT",
                      font=("Segoe UI", 12, "bold"),
                      fg_color=WARNING, text_color="#000",
                      command=self._run_diff,
                      corner_radius=6, width=120).pack(side="left", padx=10)

        ctk.CTkButton(top_frame, text="🗑️ Clear",
                      font=("Segoe UI", 11, "bold"),
                      fg_color=DANGER, text_color="#fff",
                      command=self._clear_all,
                      corner_radius=6, width=100).pack(side="right", padx=20, pady=15)

        # ── tabs ─────────────────────────────────────────────────────────────
        tab_frame = ctk.CTkFrame(self, fg_color="transparent")
        tab_frame.pack(side="top", fill="x", padx=12, pady=(12, 0))

        self.tab_var = tk.StringVar(value="results")
        for tab_name, tab_id in [("Results", "results"),
                                  ("DiffChecker", "diffchecker")]:
            ctk.CTkButton(
                tab_frame, text=tab_name,
                font=("Segoe UI", 11, "bold"),
                fg_color=ACCENT  if tab_id == "results" else BORDER,
                text_color="#000" if tab_id == "results" else TEXT_SEC,
                command=lambda tid=tab_id: self._switch_tab(tid),
                corner_radius=4, width=120,
            ).pack(side="left", padx=4)

        # ── content area ─────────────────────────────────────────────────────
        self.content_frame = ctk.CTkFrame(self, fg_color=BG_DEEP)
        self.content_frame.pack(side="top", fill="both", expand=True)

        # Results tab
        self.results_frame = ctk.CTkFrame(self.content_frame, fg_color=BG_DEEP)
        self.results_frame.pack(fill="both", expand=True)

        self.canvas1 = tk.Canvas(self.results_frame, bg="#f0f0f0",
                                  highlightthickness=0)
        self.canvas1.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        self.canvas2 = tk.Canvas(self.results_frame, bg="#f0f0f0",
                                  highlightthickness=0)
        self.canvas2.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        # DiffChecker tab
        self.dc_frame = ctk.CTkFrame(self.content_frame, fg_color=BG_DEEP)
        self.dc_frame.pack(fill="both", expand=True)

        dc_top = ctk.CTkFrame(self.dc_frame, fg_color="transparent", height=40)
        dc_top.pack(side="top", fill="x", padx=12, pady=(12, 0))
        dc_top.pack_propagate(False)

        for mode, label in [("sideby", "Side by Side"),
                             ("slider", "Slider"),
                             ("highlight", "Highlight")]:
            ctk.CTkButton(
                dc_top, text=label,
                font=("Segoe UI", 10, "bold"),
                fg_color=ACCENT  if mode == self._dc_mode else BORDER,
                text_color="#000" if mode == self._dc_mode else TEXT_SEC,
                command=lambda m=mode: self._set_dc_mode(m),
                corner_radius=4, width=100,
            ).pack(side="left", padx=4)

        ctk.CTkButton(dc_top, text="💾 Save View",
                      font=("Segoe UI", 10, "bold"),
                      fg_color=SUCCESS, text_color="#fff",
                      command=self._dc_save_view,
                      corner_radius=4, width=100).pack(side="right", padx=4)

        self.dc_canvas = tk.Canvas(self.dc_frame, bg="#f0f0f0",
                                    highlightthickness=0)
        self.dc_canvas.pack(fill="both", expand=True, padx=12, pady=12)
        self.dc_canvas.bind("<Motion>", self._on_slider_motion)

        # ── bottom log ───────────────────────────────────────────────────────
        bottom_frame = ctk.CTkFrame(self, fg_color=BG_CARD, height=120)
        bottom_frame.pack(side="bottom", fill="x")
        bottom_frame.pack_propagate(False)

        ctk.CTkLabel(bottom_frame, text="📋 Console Log",
                     font=("Segoe UI", 10, "bold"),
                     text_color=ACCENT).pack(anchor="w", padx=12, pady=(8, 4))

        self.log_text = tk.Text(
            bottom_frame, height=4,
            bg=BG_INPUT, fg=TEXT_PRI,
            font=("Courier New", 9),
            insertbackground=ACCENT,
            relief="flat", borderwidth=0,
        )
        self.log_text.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.log_text.config(state="disabled")

    # ── tab / mode switching ─────────────────────────────────────────────────

    def _switch_tab(self, tab_id: str) -> None:
        if tab_id == "results":
            self.results_frame.pack(fill="both", expand=True)
            self.dc_frame.pack_forget()
        else:
            self.results_frame.pack_forget()
            self.dc_frame.pack(fill="both", expand=True)
        self.tab_var.set(tab_id)

    def _set_dc_mode(self, mode: str) -> None:
        self._dc_mode = mode
        self._render_dc()

    def _on_slider_motion(self, event) -> None:
        if self._dc_mode == "slider":
            cw = self.dc_canvas.winfo_width()
            if cw > 0:
                self._slider_pct = max(0.0, min(1.0, event.x / cw))
                self._render_dc()

    # ── logging ──────────────────────────────────────────────────────────────

    def _log(self, message: str, level: str = "info") -> None:
        self.log_text.config(state="normal")
        colour_map = {
            "info":    TEXT_SEC,
            "success": SUCCESS,
            "warn":    WARNING,
            "error":   DANGER,
        }
        colour = colour_map.get(level, TEXT_SEC)
        self.log_text.insert("end", f"[{level.upper()}] {message}\n")
        self.log_text.tag_add(level, "end linestart", "end lineend")
        self.log_text.tag_config(level, foreground=colour)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    # ── file loading ─────────────────────────────────────────────────────────

    def _load_file_1(self) -> None:
        """Load Drawing 1 with path validation."""
        file = filedialog.askopenfilename(
            filetypes=[("CAD Files", "*.dxf *.dwg"), ("All Files", "*.*")]
        )
        if not file:
            return
        try:
            clean = _validate_file_path(file)    # ← CRITICAL
            self.path1.set(clean)
            self._log(f"Loaded Drawing 1: {os.path.basename(clean)}", "success")
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            self._log(f"Cannot load Drawing 1: {exc}", "error")
            messagebox.showerror("Invalid File", str(exc))

    def _load_file_2(self) -> None:
        """Load Drawing 2 with path validation."""
        file = filedialog.askopenfilename(
            filetypes=[("CAD Files", "*.dxf *.dwg"), ("All Files", "*.*")]
        )
        if not file:
            return
        try:
            clean = _validate_file_path(file)    # ← CRITICAL
            self.path2.set(clean)
            self._log(f"Loaded Drawing 2: {os.path.basename(clean)}", "success")
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            self._log(f"Cannot load Drawing 2: {exc}", "error")
            messagebox.showerror("Invalid File", str(exc))

    # ── diff analysis ────────────────────────────────────────────────────────

    def _run_diff(self) -> None:
        if not self.path1.get() or not self.path2.get():
            messagebox.showwarning("Missing Files", "Please load both drawings.")
            return
        thread = threading.Thread(
            target=self._diff_thread, daemon=True, name="DiffWorker"
        )
        thread.start()

    def _diff_thread(self) -> None:
        """Background worker: conversion → shape loading → matching → render.

        NICE-TO-HAVE: _AnalysisWatchdog posts a GUI warning if the thread
        exceeds ANALYSIS_TIMEOUT_S.
        """
        watchdog = _AnalysisWatchdog(self._log)
        watchdog.start()
        try:
            self._log("Converting files …", "info")
            dxf1 = prepare_dxf_path(self.path1.get(), self._log)
            dxf2 = prepare_dxf_path(self.path2.get(), self._log)

            self._log("Loading shapes …", "info")
            s1, xmin1, ymin1, xmax1, ymax1, w1, h1 = load_normalised_shapes(dxf1)
            s2, xmin2, ymin2, xmax2, ymax2, w2, h2 = load_normalised_shapes(dxf2)

            self._log("Matching shapes …", "info")
            pairs, m1, m2 = match_shapes(s1, s2)

            self._log("Rendering images …", "info")
            self.img1_pil = dxf_to_image(dxf1)
            self.img2_pil = dxf_to_image(dxf2)

            # CRITICAL: build typed RenderParams (was raw tuple p1/p2)
            rp1 = _build_render_params(xmin1, ymin1, w1, h1)
            rp2 = _build_render_params(xmin2, ymin2, w2, h2)

            hl_img = render_highlight(rp1, rp2, s1, s2, pairs, m1, m2)
            sb_img = render_side_by_side(rp1, rp2, s1, s2)   # noqa: F841

            self._dc_pil["d1"] = self.img1_pil.resize((DC_W // 2, DC_H),
                                                        PILImage.LANCZOS)
            self._dc_pil["d2"] = self.img2_pil.resize((DC_W // 2, DC_H),
                                                        PILImage.LANCZOS)
            self._dc_pil["hl"] = hl_img.resize((DC_W, DC_H), PILImage.LANCZOS)

            self._result_pil = hl_img
            self._result     = {"pairs": pairs, "m1": m1, "m2": m2}

            self._display_images()
            self._render_dc()

            self._log(
                f"Diff complete — {len(pairs)} matches, "
                f"{len(m1)} in D1, {len(m2)} in D2",
                "success",
            )

        except (ValueError, FileNotFoundError, PermissionError,
                RuntimeError, OSError) as exc:
            self._log(f"Diff error: {exc}", "error")
            messagebox.showerror("Analysis Error", str(exc))
        finally:
            watchdog.cancel()

    # ── display ──────────────────────────────────────────────────────────────

    def _display_images(self) -> None:
        if self.img1_pil:
            img = self.img1_pil.resize((600, 400), PILImage.LANCZOS)
            self._img1_tk = ImageTk.PhotoImage(img)
            self.canvas1.delete("all")
            self.canvas1.create_image(300, 200, image=self._img1_tk)

        if self.img2_pil:
            img = self.img2_pil.resize((600, 400), PILImage.LANCZOS)
            self._img2_tk = ImageTk.PhotoImage(img)
            self.canvas2.delete("all")
            self.canvas2.create_image(300, 200, image=self._img2_tk)

    def _render_dc(self) -> None:
        if not self._dc_pil.get("d1"):
            return

        CW = self.dc_canvas.winfo_width()
        CH = self.dc_canvas.winfo_height()
        if CW < 100 or CH < 100:
            self.after(100, self._render_dc)
            return

        self.dc_canvas.delete("all")
        self._dc_tk = {}

        if self._dc_mode == "sideby":
            d1, d2     = self._dc_pil["d1"], self._dc_pil["d2"]
            h1 = h2    = int(CH * 0.9)
            w1         = int(h1 * d1.width / d1.height)
            w2         = int(h2 * d2.width / d2.height)
            d1s        = d1.resize((w1, h1), PILImage.LANCZOS)
            d2s        = d2.resize((w2, h2), PILImage.LANCZOS)
            xo1        = (CW // 2 - w1) // 2
            xo2        = CW // 2 + (CW // 2 - w2) // 2
            yo         = (CH - h1) // 2
            self._dc_tk["d1"] = ImageTk.PhotoImage(d1s)
            self._dc_tk["d2"] = ImageTk.PhotoImage(d2s)
            self.dc_canvas.create_image(xo1, yo, anchor="nw",
                                         image=self._dc_tk["d1"])
            self.dc_canvas.create_image(xo2, yo, anchor="nw",
                                         image=self._dc_tk["d2"])

        elif self._dc_mode == "slider":
            d2  = self._dc_pil["d2"]
            h   = int(CH * 0.9)
            w   = int(h * d2.width / d2.height)
            d2s = d2.resize((w, h), PILImage.LANCZOS)
            xo, yo = (CW - w) // 2, (CH - h) // 2
            self._dc_tk["back"] = ImageTk.PhotoImage(d2s)
            self.dc_canvas.create_image(xo, yo, anchor="nw",
                                         image=self._dc_tk["back"])
            div_x = int(self._slider_pct * w)
            if div_x > 0:
                front = self._dc_pil["d1"].resize(
                    (w, h), PILImage.LANCZOS
                ).crop((0, 0, div_x, h))
                self._dc_tk["front"] = ImageTk.PhotoImage(front)
                self.dc_canvas.create_image(xo, yo, anchor="nw",
                                             image=self._dc_tk["front"])
            self.dc_canvas.create_line(
                xo + div_x, yo, xo + div_x, yo + h,
                fill="#00d4ff", width=3,
            )

        elif self._dc_mode == "highlight":
            hl  = self._dc_pil["hl"]
            h   = int(CH * 0.9)
            w   = int(h * hl.width / hl.height)
            hls = hl.resize((w, h), PILImage.LANCZOS)
            xo, yo = (CW - w) // 2, (CH - h) // 2
            self._dc_tk["hl"] = ImageTk.PhotoImage(hls)
            self.dc_canvas.create_image(xo, yo, anchor="nw",
                                         image=self._dc_tk["hl"])

    # ── save view ────────────────────────────────────────────────────────────

    def _dc_save_view(self) -> None:
        """Save the current DiffChecker view to an image file.

        HIGH fixes:
          • _validate_output_path() — home-dir restriction + extension check.
          • File-overwrite confirmation dialog.
        """
        if not self._dc_pil.get("hl"):
            messagebox.showwarning("No Data", "Run analysis first.")
            return

        file = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if not file:
            return

        # HIGH: validate output path (extension + home-dir)
        try:
            safe_path = _validate_output_path(file)
        except (ValueError, PermissionError) as exc:
            messagebox.showerror("Invalid Save Path", str(exc))
            return

        # HIGH: overwrite confirmation
        if os.path.isfile(safe_path):
            if not messagebox.askyesno(
                "Overwrite File?",
                f"'{os.path.basename(safe_path)}' already exists.\n\nOverwrite?",
            ):
                return

        try:
            self._dc_pil["hl"].save(safe_path)
            self._log(f"Saved → {os.path.basename(safe_path)}", "success")
        except (OSError, IOError) as exc:
            self._log(f"Save failed: {exc}", "error")
            messagebox.showerror("Save Error", str(exc))

    # ── misc ─────────────────────────────────────────────────────────────────

    def _clear_all(self) -> None:
        self.path1.set("")
        self.path2.set("")
        self.img1_pil = self.img2_pil = None
        self._dc_pil.clear()
        self.canvas1.delete("all")
        self.canvas2.delete("all")
        self.dc_canvas.delete("all")
        self._log("Cleared.", "info")

    def _check_deps(self) -> None:
        if not DXF_OK:
            self._log("❌ ezdxf not installed — pip install ezdxf", "error")
        else:
            self._log("✓ ezdxf OK", "success")
        if not SHAPELY_OK:
            self._log("❌ shapely/rtree not installed", "error")
        else:
            self._log("✓ shapely/rtree OK", "success")


# ════════════════════════════════════════════════════════════════════════════
# SECURITY TEST STUBS  (NICE-TO-HAVE — run with: pytest DrawingDiffGUI.py)
# ════════════════════════════════════════════════════════════════════════════
# Uncomment and extend when pytest is available in your environment.
#
# import pytest, pathlib
#
# class TestValidateFilePath:
#     def test_rejects_empty(self):
#         with pytest.raises(ValueError):
#             _validate_file_path("")
#
#     def test_rejects_bad_extension(self, tmp_path):
#         f = tmp_path / "drawing.txt"
#         f.write_text("bad")
#         with pytest.raises(ValueError, match="extension"):
#             _validate_file_path(str(f))
#
#     def test_rejects_missing_file(self):
#         with pytest.raises(FileNotFoundError):
#             _validate_file_path("/nonexistent/path/file.dxf")
#
#     def test_accepts_valid_dxf(self, tmp_path):
#         f = tmp_path / "ok.dxf"
#         f.write_text("DXF")
#         result = _validate_file_path(str(f))
#         assert result == str(f.resolve())
#
# class TestValidateOutputPath:
#     def test_rejects_outside_home(self, tmp_path):
#         with pytest.raises(ValueError, match="home"):
#             _validate_output_path("/etc/shadow.png")
#
#     def test_rejects_bad_extension(self, tmp_path):
#         f = tmp_path / "out.bmp"
#         with pytest.raises(ValueError, match="extension"):
#             _validate_output_path(str(f))
#
#     def test_accepts_png_in_home(self, tmp_path, monkeypatch):
#         monkeypatch.setattr("__main__.HOME_DIR", str(tmp_path.resolve()))
#         import importlib, __main__
#         __main__.HOME_DIR = str(tmp_path.resolve())
#         result = _validate_output_path(str(tmp_path / "out.png"))
#         assert result.endswith(".png")
#
# class TestCoordinateBounds:
#     def test_rejects_nan(self):
#         with pytest.raises(ValueError, match="Non-finite"):
#             _validate_coordinate_bounds(float("nan"), 0, 1, 1)
#
#     def test_rejects_inf(self):
#         with pytest.raises(ValueError, match="Non-finite"):
#             _validate_coordinate_bounds(0, float("inf"), 1, 1)
#
#     def test_rejects_out_of_range(self):
#         with pytest.raises(ValueError, match="exceeds bound"):
#             _validate_coordinate_bounds(0, 0, COORD_BOUND + 1, 1)
#
#     def test_rejects_degenerate(self):
#         with pytest.raises(ValueError, match="Degenerate"):
#             _validate_coordinate_bounds(5, 5, 5, 5)
#
#     def test_accepts_normal(self):
#         _validate_coordinate_bounds(-100, -50, 100, 50)   # must not raise
#
# class TestEntityLimit:
#     def test_entity_limit_constant(self):
#         assert ENTITY_LIMIT == 100_000
#
# class TestMakeTempDxf:
#     def test_creates_file(self):
#         path = _make_temp_dxf()
#         assert os.path.isfile(path)
#         assert path.endswith(".dxf")
#         assert path in _TEMP_FILES
#         os.remove(path)
#
#     def test_no_race_condition(self):
#         # mktemp() returns a name without creating the file; NamedTemporaryFile
#         # creates it atomically. Verify the file actually exists immediately.
#         path = _make_temp_dxf()
#         assert os.path.exists(path), "File must exist right after creation"
#         os.remove(path)
#
# class TestRenderParams:
#     def test_dxf_to_pixel_origin(self):
#         rp = RenderParams(offset_x=60, offset_y=60,
#                           xmin=0, ymin=0,
#                           scale_x=10, scale_y=10,
#                           canvas_h=1600)
#         px, py = _dxf_to_pixel(0, 0, rp)
#         assert px == 60
#         assert py == 1600 - 60   # Y-flip: origin maps to near bottom
#
#     def test_dxf_to_pixel_yflip(self):
#         rp = RenderParams(offset_x=0, offset_y=0,
#                           xmin=0, ymin=0,
#                           scale_x=1, scale_y=1,
#                           canvas_h=100)
#         _, py_low  = _dxf_to_pixel(0, 10, rp)   # higher DXF y
#         _, py_high = _dxf_to_pixel(0, 0,  rp)   # lower  DXF y
#         assert py_low < py_high, "Higher DXF y must produce smaller pixel y"


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _apply_resource_limits()     # NICE-TO-HAVE: optional Unix memory cap
    app = App()
    app.mainloop()
