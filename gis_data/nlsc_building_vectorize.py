# -*- coding: utf-8 -*-
"""
nlsc_building_vectorize.py  [v5]
=================================
QGIS Python Script - NLSC Building Vectorizer with Sub-Section Splitting
& Multi-Glyph Floor-Height OCR

WHY v5 EXISTS (empirical findings from probing real TOPO01K tiles at max
zoom 19, 2026-07-17):
  1. Floor-height labels (1R/2R/.../T/1B/2B/.../1M/2M/...) ARE legible at
     zoom 19 (0.30 m/px) to the human eye. v4's OCR failure was NOT a
     resolution problem -- it was an ALGORITHM problem: v4 ran exactly one
     OCR crop centred on each building polygon's overall centroid. Large
     buildings (e.g. shopping complexes) contain MANY independent floor-
     count cells side by side ("6R","5R","4R"...); the geometric centroid
     of the whole pink blob almost never lands on any actual text glyph,
     so OCR silently read nothing.
  2. What looked like "imprecise" building outlines in v4 was the same
     root cause from the geometry side: adjacent roof-height sections
     (roof steps, canopies, mezzanine volumes) are divided by THIN black
     lines that are easy to confuse with the diagonal HATCH texture lines
     used to fill the building interior. v4 relied on connected-component
     separation alone, which only works when the dividing line is fully
     unbroken -- in practice many are not, so sections merge.
  3. New label type confirmed in real tiles: "1M"/"2M"/... (å¤¾å±¤, mezzanine)
     alongside the previously-known R (åœ°ä¸Šæ¨“å±¤), T (ä¸€æ¨“/åœ°é¢å±¤), and now
     also confirmed B (åœ°ä¸‹å®¤/B1,B2...). Regex extended to all four.

ALGORITHM CHANGE (the actual fix):
  A. Pink building mask, same as v4 (colour-exact, no bias).
  B. A SEPARATE "black ink" mask captures both the outer border AND the
     interior hatch/dividing lines together.
  C. Morphological opening with a small kernel (~hatch line width) strips
     out the thin diagonal hatch strokes while preserving thicker, more
     continuous dividing/border lines -- hatch lines are short, thin,
     repeating segments; true section borders are thicker and/or longer
     unbroken runs. This is the same "thick border vs thin fill" concept
     the user specified, made algorithmic via cv2 morphologyEx(OPEN).
  D. Border-flood-fill (same technique already used in v4 for the B5000
     confirmation mask) is applied to this CLEANED border network to
     carve each pink blob into its independently-enclosed sub-faces --
     one sub-face = one roof-height section = one output polygon.
  E. Floor-height OCR no longer runs once per building centroid. It scans
     the ENTIRE interior of each SUB-FACE for black-glyph connected
     components (filtered by size/aspect-ratio to exclude hatch strokes),
     and OCRs each candidate glyph cluster directly -- so a label is
     found wherever it actually sits, not where the centroid happens to be.
  F. A rough 3D-ready height estimate (est_height_m) is attached per
     sub-face: floors_above * FLOOR_HEIGHT_M, minus basement handling.
     This is a MODELLING ASSUMPTION (uniform floor height), not a
     measured value -- flagged honestly, see attribute docstring below.

Usage: same as v4 -- select a polygon extent layer in QGIS, run this file
from Plugins > Python Console > Show Editor.

Attributes:
    floor_text     - raw label matched, e.g. "6R", "2B", "1M", "T"
    floor_kind     - "R" (above-ground) | "T" (ground) | "B" (basement) | "M" (mezzanine)
    floor_number   - integer level within its kind (T -> 1)
    est_height_m   - floors_above * FLOOR_HEIGHT_M (0 if basement-only or unknown)
    confidence     - "high"/"medium"/"low"/"unverified" (3-service cross-validation, as v4)
    b5000_overlap, emap_overlap - as v4
    sub_id         - integer sub-face id within its parent pink blob (0 if not split)
"""

import sys, os, io, re, warnings
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---- User-configurable -------------------------------------------------------
ZOOM           = 19     # TOPO01K WMTS max available zoom (0.2986 m/px) -- confirmed
                         # via GetCapabilities 2026-07-17; do not raise, server has no
                         # tiles beyond 19 and will silently 404 / return blank.
BUFFER_TILES   = 1
MIN_AREA_M2    = 2.0
MIN_SUBFACE_AREA_M2 = 1.5  # drop sub-faces smaller than this -- empirically,
                            # anti-aliasing residue after hatch-stripping still
                            # produces many few-pixel noise fragments even at
                            # 8-connectivity; real roof-height sections are
                            # never this small (confirmed 2026-07-17 on real data)
SIMPLIFY_M     = 0.25    # tighter than v4 (was 0.3) -- zoom 19 supports finer detail
FORCE_LAYER_ID = ""
OUT_LAYER_NAME = "buildings_nlsc"
FLOOR_OCR      = True
OVERLAP_THRESH = 0.50
FLOOR_HEIGHT_M = 3.6     # site_constraints.yaml regulations.floor_height default
DEBUG_SAVE_CROPS = False # set True to dump every OCR glyph-crop to gis_data/ocr_debug/
                         # for visual troubleshooting if OCR is still unreliable

# Morphological opening kernel size (pixels) used to strip hatch lines while
# keeping true dividing borders. Hatch strokes in TOPO01K render ~1px wide;
# true section borders are typically 2px+ or form long unbroken runs.
HATCH_STRIP_KERNEL_PX = 2

WMTS_ROOT     = "https://wmts.nlsc.gov.tw/wmts"
WMTS_CAPS_URL = ("https://maps.nlsc.gov.tw/wmtsTOPO01K/"
                 "?SERVICE=WMTS&REQUEST=GetCapabilities")
TILE_URL_TMPL = WMTS_ROOT + "/{layer}/default/GoogleMapsCompatible/{z}/{y}/{x}"

WMS_URL        = "https://wms.nlsc.gov.tw/wms"
WMS_LAYER_B5000 = "B5000"
WMS_LAYER_EMAP  = "EMAP"

BLDG_R, BLDG_G, BLDG_B = 255, 117, 117
EMAP_BUILDING_RGB      = (234, 227, 234)
EMAP_BUILDING_TOL      = 8
BLACK_INK_MAXRGB       = 90   # near-black threshold; includes anti-aliased greys

MAX_TILES = 100

_SCRIPT_DIR = (os.path.dirname(os.path.abspath(__file__))
               if "__file__" in dir() else os.getcwd())
_DEBUG_DIR = os.path.join(_SCRIPT_DIR, "ocr_debug")

# ---- Core dependencies -------------------------------------------------------
try:
    import requests
    from PIL import Image
    import numpy as np
    from scipy import ndimage as ndi
    print("[OK] requests / PIL / numpy / scipy loaded")
except ImportError as e:
    raise ImportError(f"Missing: {e}\nRun: pip install requests pillow numpy scipy")

_CV2 = False
try:
    import cv2 as _cv2
    _CV2 = True
    print("[OK] OpenCV (cv2) -- morphological hatch-stripping + TC89 contours enabled")
except ImportError:
    print("[ERROR] cv2 is REQUIRED for v5's hatch/border separation "
          "(cv2.morphologyEx). Run in OSGeo4W Shell: python -m pip install opencv-python")

_SKIMAGE = False
_ski_find = None
_ski_approx = None
try:
    from skimage.measure import find_contours as _ski_find
    try:
        from skimage.measure import approximate_polygon as _ski_approx
    except ImportError:
        pass
    _SKIMAGE = True
except ImportError:
    pass

_OCR_AVAILABLE = False
_OCR_DIAG = ""
_tess = None
try:
    import pytesseract as _tess
    for _tp in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
        if os.path.exists(_tp):
            _tess.pytesseract.tesseract_cmd = _tp
            break
    try:
        _ver = _tess.get_tesseract_version()
        _OCR_AVAILABLE = True
        print(f"[OK] pytesseract bound to Tesseract {_ver} -- floor OCR enabled")
    except Exception as e:
        _OCR_DIAG = (f"pytesseract imported OK but the Tesseract binary call failed: "
                     f"{e}. Check the _tp paths list matches your install.")
        print(f"[WARN] {_OCR_DIAG}")
except ImportError:
    _OCR_DIAG = (
        "pytesseract is NOT installed in QGIS's Python environment. This is "
        "almost always a DIFFERENT python.exe than your system Python. Fix: "
        "1) In QGIS Python Console run:  import sys; print(sys.executable)  "
        "to find which interpreter QGIS uses.\n"
        "2) Open OSGeo4W Shell (Start Menu, NOT a normal terminal) and run: "
        "python -m pip install pytesseract\n"
        "3) If step 2's python is not the SAME one printed in step 1, locate "
        "it manually (usually C:\\Program Files\\QGIS 3.xx\\apps\\Python3x\\python.exe) "
        "and run:  <that path> -m pip install pytesseract")
    print(f"[WARN] {_OCR_DIAG}")

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry,
    QgsPointXY, QgsField, QgsFields, QgsCoordinateReferenceSystem,
    QgsCoordinateTransform, QgsRectangle, QgsWkbTypes,
    QgsSymbol, QgsSimpleFillSymbolLayer, QgsSingleSymbolRenderer,
    QgsCategorizedSymbolRenderer, QgsRendererCategory,
)
from PyQt5.QtCore import QVariant
from PyQt5.QtGui import QColor


# =============================================================================
# 1. Tile math
# =============================================================================
_ORIGIN     = -20037508.34278925
_WORLD_SIZE =  40075016.68557849

def _tile_size_m(z):
    return _WORLD_SIZE / (2 ** z)

def _3857_to_tile(mx, my, z):
    ts = _tile_size_m(z)
    return int((mx - _ORIGIN) / ts), int((-_ORIGIN - my) / ts)

def _pixel_size_m(z):
    return _tile_size_m(z) / 256


# =============================================================================
# 2. WMTS layer auto-detect
# =============================================================================
_detected_layer_id = [None]

def _find_layer_id(ext_3857, force=""):
    if force:
        return force
    if _detected_layer_id[0]:
        return _detected_layer_id[0]
    try:
        from xml.etree import ElementTree as ET
        r = requests.get(WMTS_CAPS_URL, timeout=15, verify=False)
        root = ET.fromstring(r.text)
        ns = {"wmts": "http://www.opengis.net/wmts/1.0",
              "ows":  "http://www.opengis.net/ows/1.1"}
        cx = (ext_3857.xMinimum() + ext_3857.xMaximum()) / 2
        cy = (ext_3857.yMinimum() + ext_3857.yMaximum()) / 2
        tr = QgsCoordinateTransform(
            QgsCoordinateReferenceSystem("EPSG:3857"),
            QgsCoordinateReferenceSystem("EPSG:4326"),
            QgsProject.instance())
        pt = tr.transform(QgsPointXY(cx, cy))
        lon_c, lat_c = pt.x(), pt.y()
        for lyr in root.findall(".//wmts:Layer", ns):
            ident = lyr.findtext("ows:Identifier", "", ns)
            bb = (lyr.find(".//ows:BoundingBox", ns) or
                  lyr.find(".//ows:WGS84BoundingBox", ns))
            if bb is not None:
                lo = bb.findtext("ows:LowerCorner", "", ns).split()
                hi = bb.findtext("ows:UpperCorner", "", ns).split()
                if (float(lo[0]) <= lon_c <= float(hi[0]) and
                        float(lo[1]) <= lat_c <= float(hi[1])):
                    print(f"[INFO] Auto-detected WMTS layer: {ident}")
                    _detected_layer_id[0] = ident
                    return ident
    except Exception as e:
        print(f"[WARN] Layer auto-detect failed: {e}")
    fallback = "TOPO01K_O01"
    print(f"[INFO] Falling back to layer: {fallback}")
    _detected_layer_id[0] = fallback
    return fallback


# =============================================================================
# 3. Tile fetch
# =============================================================================
def _choose_zoom(ext_3857, desired_zoom, buf):
    z = desired_zoom
    while z > 14:
        tx0, ty0 = _3857_to_tile(ext_3857.xMinimum(), ext_3857.yMaximum(), z)
        tx1, ty1 = _3857_to_tile(ext_3857.xMaximum(), ext_3857.yMinimum(), z)
        tx0 -= buf; ty0 -= buf; tx1 += buf; ty1 += buf
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= MAX_TILES:
            return z, tx0, ty0, tx1, ty1
        z -= 1
    return z, tx0, ty0, tx1, ty1


def fetch_topo01k_mosaic(ext_3857, layer_id, desired_zoom=ZOOM, buf=BUFFER_TILES):
    z, tx0, ty0, tx1, ty1 = _choose_zoom(ext_3857, desired_zoom, buf)
    ncols = tx1 - tx0 + 1
    nrows = ty1 - ty0 + 1
    print(f"    [TOPO01K] fetch {ncols}x{nrows} tiles @ z={z} "
          f"({_pixel_size_m(z):.3f} m/px) ...")

    mosaic = Image.new("RGBA", (ncols * 256, nrows * 256), (255, 255, 255, 0))
    sess = requests.Session(); sess.verify = False
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            url = TILE_URL_TMPL.format(layer=layer_id, z=z, y=ty, x=tx)
            try:
                resp = sess.get(url, timeout=20)
                if resp.status_code == 200 and resp.content[:4] == b"\x89PNG":
                    tile = Image.open(io.BytesIO(resp.content)).convert("RGBA")
                    mosaic.paste(tile, ((tx - tx0) * 256, (ty - ty0) * 256))
            except Exception:
                pass

    ts   = _tile_size_m(z)
    xmin = _ORIGIN + tx0 * ts
    ymax = -_ORIGIN - ty0 * ts
    px_m = ts / 256
    return np.array(mosaic), xmin, ymax, px_m


def fetch_wms_aligned(mosaic_xmin, mosaic_ymax, px_m, width, height, layer_name):
    xmin = mosaic_xmin
    ymax = mosaic_ymax
    xmax = xmin + width  * px_m
    ymin = ymax - height * px_m
    bbox = f"{xmin},{ymin},{xmax},{ymax}"
    params = {
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
        "LAYERS": layer_name, "STYLES": "", "SRS": "EPSG:3857",
        "BBOX": bbox, "WIDTH": str(width), "HEIGHT": str(height),
        "FORMAT": "image/png", "TRANSPARENT": "TRUE",
    }
    try:
        resp = requests.get(WMS_URL, params=params, timeout=30, verify=False)
        if resp.status_code == 200 and resp.content[:4] == b"\x89PNG":
            img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            out = Image.alpha_composite(bg, img).convert("RGB")
            return np.array(out)
        else:
            print(f"    [WARN] WMS {layer_name}: HTTP {resp.status_code} or non-PNG")
    except Exception as e:
        print(f"    [WARN] WMS {layer_name} fetch failed: {e}")
    return None


# =============================================================================
# 4. Masks: pink fill, black ink (border+hatch combined), confirmation sources
# =============================================================================
def extract_building_mask(mosaic_rgba):
    """TOPO01K pink fill (255,117,117)."""
    r, g, b, a = (mosaic_rgba[:, :, i] for i in range(4))
    return (r == BLDG_R) & (g == BLDG_G) & (b == BLDG_B) & (a > 200)


def extract_black_ink_mask(mosaic_rgba):
    """All near-black ink in TOPO01K: outer border + interior hatch + dividing
    lines + floor-height text, all mixed together at this stage."""
    r, g, b, a = (mosaic_rgba[:, :, i] for i in range(4))
    return (r < BLACK_INK_MAXRGB) & (g < BLACK_INK_MAXRGB) & (b < BLACK_INK_MAXRGB) & (a > 100)


# v3/v4 identified building "blobs" as raw connected pink pixels, on the
# assumption that internal black lines would naturally split adjacent
# sections. Empirically probing a real zoom-19 TOPO01K tile (2026-07-17)
# disproved this: the diagonal hatch texture overlaid on the pink fill
# chops it into thousands of tiny 1-a-few-pixel fragments (16030 components,
# largest only 224px, on a 768x768 test tile) -- there is no single
# "building-scale" pink blob to find contours from in the first place.
#
# Fix: hatch strokes are BLACK ink physically touching the pink fill on
# both sides, so `pink | black_ink` recombines the true building silhouette
# without any morphological closing (which would distort/over-merge the
# boundary -- iterations=2 closing alone inflated total area 3.8x and still
# didn't fully recover single blobs). The combined mask recovered 715
# building-scale components (largest 19136px) on the same test tile with
# ZERO added pixels beyond what pink+ink actually cover.
def extract_building_extent_mask(mosaic_rgba):
    """pink OR black-ink pixels -- the true building silhouette including
    hatch-fragmented fill, recovered without any distorting morphology."""
    return extract_building_mask(mosaic_rgba) | extract_black_ink_mask(mosaic_rgba)


def strip_hatch_keep_borders(black_ink_mask):
    """
    Morphological opening: erode then dilate with a small kernel. Thin,
    short hatch strokes (~1px) vanish under erosion and don't come back;
    thicker/longer dividing-border line segments survive. This is how we
    algorithmically implement the user's distinction: thick border=division,
    thin hatch=fill -- instead of relying on connected-component luck.
    """
    if not _CV2:
        return black_ink_mask   # degraded: no stripping possible
    k = max(1, HATCH_STRIP_KERNEL_PX)
    kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (k, k))
    opened = _cv2.morphologyEx(black_ink_mask.astype(np.uint8) * 255,
                                _cv2.MORPH_OPEN, kernel)
    return opened > 0


def build_b5000_confirmation_mask(b5000_rgb):
    if b5000_rgb is None:
        return None
    r, g, b = b5000_rgb[:, :, 0], b5000_rgb[:, :, 1], b5000_rgb[:, :, 2]
    is_line = ~((r > 240) & (g > 240) & (b > 240))
    free = ~is_line
    labeled, n = ndi.label(free)
    if n == 0:
        return np.zeros(b5000_rgb.shape[:2], dtype=bool)
    border_labels = set(labeled[0, :]) | set(labeled[-1, :]) \
                  | set(labeled[:, 0]) | set(labeled[:, -1])
    border_labels.discard(0)
    exterior = np.isin(labeled, list(border_labels))
    return free & ~exterior


def build_emap_confirmation_mask(emap_rgb):
    if emap_rgb is None:
        return None
    tr, tg, tb = EMAP_BUILDING_RGB
    r = emap_rgb[:, :, 0].astype(int)
    g = emap_rgb[:, :, 1].astype(int)
    b = emap_rgb[:, :, 2].astype(int)
    return ((np.abs(r - tr) <= EMAP_BUILDING_TOL) &
            (np.abs(g - tg) <= EMAP_BUILDING_TOL) &
            (np.abs(b - tb) <= EMAP_BUILDING_TOL))


# =============================================================================
# 5. Sub-face splitting: building-extent blob -> independent height-section polygons
# =============================================================================
def split_blob_into_subfaces(extent_comp, black_ink_full):
    """
    extent_comp    : bool mask, one connected (pink|ink) building-extent
                      component, already isolated to this blob's bbox
    black_ink_full : bool mask (same bbox) of ALL black ink in this crop

    Returns a list of bool masks, each one enclosed sub-face (candidate
    independent building / roof-height section) within extent_comp.
    """
    border_net = strip_hatch_keep_borders(black_ink_full)
    # A pixel belongs to "free interior space" if it's building extent and
    # not itself a (thick, non-hatch) border-network pixel.
    free = extent_comp & ~border_net
    # 8-connectivity is required here, not 4: scattered single-pixel ink
    # residue left after hatch-stripping (anti-aliasing remnants) breaks
    # diagonal adjacency under 4-connectivity, fracturing one real section
    # into thousands of spurious 1-a-few-pixel "sub-faces" (empirically
    # confirmed 2026-07-17: 9403 fake sub-faces at 4-connectivity collapsed
    # to 125 plausible ones, including a correctly-recovered 15162px main
    # section, once switched to 8-connectivity).
    labeled, n = ndi.label(free, structure=ndi.generate_binary_structure(2, 2))
    if n <= 1:
        return [extent_comp] if extent_comp.any() else []

    subfaces = []
    for lbl in range(1, n + 1):
        face = (labeled == lbl)
        # Re-absorb a thin ring of adjacent border pixels into the face so
        # traced contours meet at shared edges instead of leaving gaps.
        face_grown = ndi.binary_dilation(face, iterations=1) & extent_comp
        subfaces.append(face_grown)
    return subfaces


# =============================================================================
# 6. Sub-pixel contour extraction
# =============================================================================
def _contour_cv2(patch_u8, tol_px):
    contours, _ = _cv2.findContours(
        patch_u8, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return None
    c = max(contours, key=_cv2.contourArea)
    approx = _cv2.approxPolyDP(c, tol_px, closed=True)
    return [(int(pt[0][1]), int(pt[0][0])) for pt in approx]


def _contour_skimage(patch_bool, tol_px):
    contours = _ski_find(patch_bool.astype(float), 0.5)
    if not contours:
        return None
    c = max(contours, key=len)
    if _ski_approx is not None:
        c = _ski_approx(c, tolerance=tol_px)
    return [(float(row), float(col)) for row, col in c]


def _contour_fallback(patch_bool):
    border = patch_bool & ~ndi.binary_erosion(patch_bool)
    rows, cols = np.where(border)
    if len(rows) < 3:
        return None
    cy, cx = rows.mean(), cols.mean()
    order = np.argsort(np.arctan2(rows - cy, cols - cx))
    return list(zip(rows[order].tolist(), cols[order].tolist()))


def trace_contour(patch_bool, tol_px):
    if _CV2:
        return _contour_cv2((patch_bool.astype(np.uint8)) * 255, tol_px)
    elif _SKIMAGE:
        return _contour_skimage(patch_bool, tol_px)
    return _contour_fallback(patch_bool)


# =============================================================================
# 7. Multi-glyph floor-height OCR (scans whole sub-face interior)
# =============================================================================
_FLOOR_RE = re.compile(r"^(\d{1,2})([RrTtBbMm])$|^([Tt])$")

def _parse_floor(text):
    """Returns (floor_text, kind, number) or (None, None, None)."""
    text = text.strip().upper()
    if text == "T":
        return "T", "T", 1
    m = re.match(r"^(\d{1,2})([RTBM])$", text)
    if not m:
        return None, None, None
    num, kind = m.group(1), m.group(2)
    return f"{num}{kind}", kind, int(num)


def _find_glyph_clusters(black_ink_mask, min_px=3, max_px=180):
    """
    Find candidate text-glyph connected components within a black-ink mask.
    Filters by pixel count and bounding-box aspect ratio to exclude long
    thin hatch strokes (aspect ratio far from square) and keep compact
    blob clusters (character strokes cluster tightly for small map labels).
    """
    labeled, n = ndi.label(black_ink_mask, structure=ndi.generate_binary_structure(2, 2))
    # ndi.find_objects requires an INTEGER-labelled array (bool masks raise
    # TypeError on newer scipy) -- use the already-integer `labeled` array
    # and index by component id, not by passing a bool mask back in.
    slices = ndi.find_objects(labeled)
    clusters = []
    for lbl in range(1, n + 1):
        sl = slices[lbl - 1]
        if sl is None:
            continue
        npx = int((labeled[sl] == lbl).sum())
        if npx < min_px or npx > max_px:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if h == 0 or w == 0:
            continue
        aspect = max(h, w) / max(1, min(h, w))
        if aspect > 6:   # hatch strokes are long & thin; glyph strokes aren't
            continue
        cy = (sl[0].start + sl[0].stop) // 2
        cx = (sl[1].start + sl[1].stop) // 2
        clusters.append((cx, cy))
    return clusters


def _merge_nearby(points, radius_px):
    """Merge glyph-stroke clusters that sit within radius_px of each other
    into single label candidates (a 2-character label like '6R' is 2+
    separate glyph blobs)."""
    if not points:
        return []
    pts = list(points)
    merged = []
    used = [False] * len(pts)
    for i, (x0, y0) in enumerate(pts):
        if used[i]:
            continue
        group = [(x0, y0)]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j, (x1, y1) in enumerate(pts):
                if used[j]:
                    continue
                if any(((x1 - gx) ** 2 + (y1 - gy) ** 2) ** 0.5 <= radius_px
                       for gx, gy in group):
                    group.append((x1, y1))
                    used[j] = True
                    changed = True
        gx = sum(p[0] for p in group) / len(group)
        gy = sum(p[1] for p in group) / len(group)
        merged.append((gx, gy))
    return merged


def ocr_subface_labels(mosaic_rgba, black_ink_mask, subface_mask,
                        offset_xy, px_m, diag_log=None):
    """
    Scans the interior of ONE sub-face for floor-height text, trying every
    glyph-cluster location found rather than a single fixed centroid crop.
    Returns list of (floor_text, kind, number) -- usually 0 or 1 entries.
    """
    if not (_OCR_AVAILABLE and FLOOR_OCR):
        if diag_log is not None:
            diag_log.append("ocr_unavailable")
        return []

    local_ink = black_ink_mask & subface_mask
    glyph_px = _find_glyph_clusters(local_ink)
    label_candidates = _merge_nearby(glyph_px, radius_px=max(3, int(0.6 / px_m)))

    if not label_candidates:
        if diag_log is not None:
            diag_log.append("no_glyph_clusters_in_subface")
        return []

    found = []
    ox, oy = offset_xy
    for (cx, cy) in label_candidates:
        radius = max(14, int(0.9 / px_m))
        crop_result = _ocr_crop_at(mosaic_rgba, int(cx + ox), int(cy + oy), radius)
        if crop_result:
            floor_text, kind, num = crop_result
            found.append((floor_text, kind, num))
            if DEBUG_SAVE_CROPS:
                _save_debug_crop(mosaic_rgba, int(cx + ox), int(cy + oy), radius, floor_text)

    if not found and diag_log is not None:
        diag_log.append(f"glyph_clusters_found_{len(label_candidates)}_but_no_regex_match")
    return found


def _ocr_crop_at(mosaic_rgba, cx_px, cy_px, radius):
    H, W = mosaic_rgba.shape[:2]
    x0, x1 = max(0, cx_px - radius), min(W, cx_px + radius)
    y0, y1 = max(0, cy_px - radius), min(H, cy_px + radius)
    crop = mosaic_rgba[y0:y1, x0:x1, :3]
    if crop.size == 0:
        return None
    img = Image.fromarray(crop)
    scale = max(1, 160 // max(img.height, 1))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    arr = np.array(img)
    gray = arr[:, :, :3].mean(axis=2)
    dark_thresh = max(60, np.percentile(gray, 20))
    is_text = gray <= dark_thresh
    bw = np.where(is_text[:, :, None], 0, 255).astype(np.uint8)
    bw_img = Image.fromarray(np.repeat(bw, 3, axis=2))

    # Single-character results (esp. "T") are far more prone to being noise
    # misread than 2-character results ("3R","1B","2M") -- random hatch/
    # anti-aliasing residue can plausibly resemble one stroke but rarely two
    # coherent characters in sequence. Empirically, psm=10 (single-char mode)
    # alone produced dozens of spurious "T" hits on tiny noise fragments
    # (2026-07-17 test). Fix: accept multi-char matches on first hit, but
    # require "T" to be confirmed by at least 2 independent PSM modes.
    t_votes = 0
    for psm in (8, 7, 10, 6):
        try:
            raw = _tess.image_to_string(
                bw_img,
                config=f"--psm {psm} -c tessedit_char_whitelist=0123456789RrTtBbMm"
            ).strip()
        except Exception:
            continue
        if not raw:
            continue
        ft, kind, num = _parse_floor(raw)
        if not ft:
            continue
        if len(ft) >= 2:
            return ft, kind, num
        if ft == "T":
            t_votes += 1
            if t_votes >= 2:
                return ft, kind, num
    return None


def _save_debug_crop(mosaic_rgba, cx_px, cy_px, radius, label_hint):
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    H, W = mosaic_rgba.shape[:2]
    x0, x1 = max(0, cx_px - radius), min(W, cx_px + radius)
    y0, y1 = max(0, cy_px - radius), min(H, cy_px + radius)
    crop = mosaic_rgba[y0:y1, x0:x1, :3]
    if crop.size == 0:
        return
    fname = f"{label_hint or 'nomatch'}_{cx_px}_{cy_px}.png"
    Image.fromarray(crop).save(os.path.join(_DEBUG_DIR, fname))


# =============================================================================
# 8. Height estimate
# =============================================================================
def estimate_height_m(labels_found):
    """
    labels_found: list of (floor_text, kind, number) collected for one sub-face.
    Uses the MAXIMUM above-ground R/T level found (top floor) as the building
    height driver; basement (B) and mezzanine (M) levels do not by themselves
    raise the above-ground envelope height in this estimate.
    NOTE: this is a modelling simplification for downstream 3D massing, not
    a surveyed value -- floors are assumed uniform FLOOR_HEIGHT_M.
    """
    above_ground = [num for (_, kind, num) in labels_found if kind in ("R", "T")]
    if not above_ground:
        return 0.0
    return max(above_ground) * FLOOR_HEIGHT_M


# =============================================================================
# 9. Per-blob processing: mask -> sub-faces -> contours -> OCR -> confidence
# =============================================================================
def process_extent_blob(comp, pink_mask, black_ink_mask, mosaic,
                         mosaic_xmin, mosaic_ymax, px_m,
                         b5000_mask, emap_mask, tol_px):
    """
    comp: bool mask of one connected (pink|ink) building-extent component
          (full mosaic-sized array).
    Returns list of dicts: geom, confidence, b5000_ovl, emap_ovl, labels, sub_id
    """
    # bbox via np.where -- ndi.find_objects requires an int-labelled array
    # and raises TypeError on a plain bool mask (confirmed on scipy in this
    # environment; do not pass `comp` directly to find_objects).
    rows, cols = np.where(comp)
    if len(rows) == 0:
        return []
    r0 = max(int(rows.min()) - 2, 0)
    r1 = min(int(rows.max()) + 3, comp.shape[0])
    c0 = max(int(cols.min()) - 2, 0)
    c1 = min(int(cols.max()) + 3, comp.shape[1])

    extent_crop = comp[r0:r1, c0:c1]
    ink_crop    = black_ink_mask[r0:r1, c0:c1]
    pink_crop   = pink_mask[r0:r1, c0:c1]

    subfaces = split_blob_into_subfaces(extent_crop, ink_crop)

    min_subface_px = max(4, int(MIN_SUBFACE_AREA_M2 / (px_m ** 2)))
    results = []
    for sub_id, face in enumerate(subfaces):
        npx = int(face.sum())
        if npx < min_subface_px:
            continue
        # Reject sub-faces with essentially no pink content -- these are
        # ink-only fragments (a stray road/contour-line segment that got
        # pulled into the building-extent mask by adjacency) rather than
        # a real building section. Real sections are pink fill + ink hatch;
        # pure ink with near-zero pink is never an actual building face.
        if (face & pink_crop).sum() < max(2, 0.05 * npx):
            continue

        rc = trace_contour(face, tol_px)
        if rc is None or len(rc) < 3:
            continue

        pts = []
        for (row, col) in rc:
            mx = mosaic_xmin + (c0 + col + 0.5) * px_m
            my = mosaic_ymax - (r0 + row + 0.5) * px_m
            pts.append(QgsPointXY(mx, my))
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        geom = QgsGeometry.fromPolygonXY([pts])
        if geom is None or geom.isEmpty():
            continue

        # Confidence cross-validation against B5000/EMAP -- map crop-local
        # face mask back to full-mosaic coordinates for the overlap test.
        full_face = np.zeros_like(comp)
        full_face[r0:r1, c0:c1] = face
        b5000_ovl = -1.0
        emap_ovl  = -1.0
        if b5000_mask is not None:
            b5000_ovl = float((full_face & b5000_mask).sum()) / npx
        if emap_mask is not None:
            emap_ovl = float((full_face & emap_mask).sum()) / npx
        votes, n_sources = 0, 0
        if b5000_ovl >= 0:
            n_sources += 1; votes += int(b5000_ovl >= OVERLAP_THRESH)
        if emap_ovl >= 0:
            n_sources += 1; votes += int(emap_ovl >= OVERLAP_THRESH)
        if n_sources == 0:
            confidence = "unverified"
        elif votes == n_sources:
            confidence = "high"
        elif votes > 0:
            confidence = "medium"
        else:
            confidence = "low"

        diag_log = []
        labels_found = ocr_subface_labels(
            mosaic, black_ink_mask, full_face, (0, 0), px_m, diag_log=diag_log)

        results.append({
            "geom": geom, "confidence": confidence,
            "b5000_overlap": round(b5000_ovl, 3), "emap_overlap": round(emap_ovl, 3),
            "labels": labels_found, "sub_id": sub_id, "diag": diag_log,
        })
    return results


# =============================================================================
# 10. Main
# =============================================================================
def main():
    print("=" * 70)
    print("NLSC Building Vectorizer  [v5 -- sub-section split + multi-glyph OCR]")
    print("=" * 70)

    if not _OCR_AVAILABLE:
        print(f"\n[DIAGNOSTIC] Floor-height OCR is DISABLED this run.\n{_OCR_DIAG}\n")
    if not _CV2:
        print("\n[DIAGNOSTIC] cv2 missing -- hatch/border separation degraded to "
              "no-op, sub-section splitting will under-perform. Install opencv-python "
              "in QGIS's Python environment.\n")

    extent_layer = iface.activeLayer()
    if extent_layer is None:
        print("[ERROR] No layer selected. Click a polygon layer in the Layers Panel first.")
        return
    if not hasattr(extent_layer, "getFeatures"):
        print("[ERROR] Selected layer is not a vector layer.")
        return

    src_crs  = extent_layer.crs()
    crs_3857 = QgsCoordinateReferenceSystem("EPSG:3857")
    tr_fwd   = QgsCoordinateTransform(src_crs, crs_3857, QgsProject.instance())

    feats = []
    for feat in extent_layer.getFeatures():
        g = feat.geometry()
        if g is None or g.isEmpty():
            continue
        g3857 = QgsGeometry(g)
        g3857.transform(tr_fwd)
        feats.append(g3857)
    if not feats:
        print("[ERROR] Extent layer has no polygon features.")
        return

    union_g = feats[0]
    for g in feats[1:]:
        union_g = union_g.combine(g)
    global_ext = union_g.boundingBox()
    print(f"[INFO] Extent layer : {extent_layer.name()}  ({len(feats)} feature(s))")

    layer_id = _find_layer_id(global_ext, FORCE_LAYER_ID)

    mem_layer = QgsVectorLayer("Polygon?crs=EPSG:3857", OUT_LAYER_NAME, "memory")
    pr = mem_layer.dataProvider()
    fields = QgsFields()
    fields.append(QgsField("floor_text",   QVariant.String))
    fields.append(QgsField("floor_kind",   QVariant.String))
    fields.append(QgsField("floor_number", QVariant.Int))
    fields.append(QgsField("est_height_m", QVariant.Double))
    fields.append(QgsField("confidence",   QVariant.String))
    fields.append(QgsField("b5000_overlap",QVariant.Double))
    fields.append(QgsField("emap_overlap", QVariant.Double))
    fields.append(QgsField("sub_id",       QVariant.Int))
    pr.addAttributes(fields)
    mem_layer.updateFields()

    total_added = 0
    stats = {"high": 0, "medium": 0, "low": 0, "unverified": 0}
    ocr_success = 0
    all_diag = []

    for fidx, clip_g in enumerate(feats, 1):
        feat_ext = clip_g.boundingBox()
        print(f"\n[Feature {fidx}/{len(feats)}]")

        mosaic, xmin, ymax, px_m = fetch_topo01k_mosaic(
            feat_ext, layer_id, ZOOM, BUFFER_TILES)
        H, W = mosaic.shape[0], mosaic.shape[1]
        print(f"    TOPO01K mosaic {W}x{H} px  pixel={px_m:.4f} m")

        b5000_rgb = fetch_wms_aligned(xmin, ymax, px_m, W, H, WMS_LAYER_B5000)
        emap_rgb  = fetch_wms_aligned(xmin, ymax, px_m, W, H, WMS_LAYER_EMAP)
        b5000_mask = build_b5000_confirmation_mask(b5000_rgb)
        emap_mask  = build_emap_confirmation_mask(emap_rgb)

        pink_mask   = extract_building_mask(mosaic)
        ink_mask    = extract_black_ink_mask(mosaic)
        extent_mask = pink_mask | ink_mask   # see extract_building_extent_mask
                                              # docstring for why raw pink alone
                                              # fragments into thousands of tiny
                                              # hatch-separated islands at zoom 19
        labeled, n = ndi.label(extent_mask, structure=ndi.generate_binary_structure(2, 2))
        min_area_px = max(4, int(MIN_AREA_M2 / (px_m ** 2)))
        tol_px = max(1.0, SIMPLIFY_M / px_m)
        print(f"    building-extent components (pink|ink): {n}")

        added_this = 0
        for blob_id in range(1, n + 1):
            comp = (labeled == blob_id)
            if comp.sum() < min_area_px:
                continue
            # A blob must contain a meaningful amount of actual pink fill to
            # be a candidate building at all -- rejects ink-only fragments
            # (road lines, contour lines, text) that happen to be spatially
            # isolated enough to form their own connected component.
            if (comp & pink_mask).sum() < max(4, min_area_px * 0.1):
                continue
            blob_results = process_extent_blob(
                comp, pink_mask, ink_mask, mosaic, xmin, ymax, px_m,
                b5000_mask, emap_mask, tol_px)

            for res in blob_results:
                clipped = res["geom"].intersection(clip_g)
                if clipped is None or clipped.isEmpty():
                    continue
                labels = res["labels"]
                # De-duplicate identical (kind, number) hits from multiple
                # glyph clusters/PSM votes -- these are the SAME real label
                # found more than once, not distinct floors.
                distinct = sorted(set((k, n_) for (_, k, n_) in labels))
                if len(distinct) == 1:
                    kind, num = distinct[0]
                    floor_text = ("T" if kind == "T" else f"{num}{kind}")
                    ocr_success += 1
                elif len(distinct) > 1:
                    # HONESTY: this sub-face genuinely contains multiple
                    # different floor labels -- almost always means the
                    # hatch/border split failed to separate what should be
                    # independent sections (thick-vs-thin line heuristic
                    # under-performs on orthogonal grid dividers that are
                    # the same width as the diagonal hatch strokes; known
                    # limitation, see continuation plan). Record ALL found
                    # labels rather than silently keeping only the first.
                    floor_text = "AMBIGUOUS:" + ";".join(
                        ("T" if k == "T" else f"{n_}{k}") for k, n_ in distinct)
                    kind, num = "AMBIGUOUS", 0
                    ocr_success += 1
                else:
                    floor_text, kind, num = "", "", 0
                    all_diag.extend(res["diag"] or ["unknown"])
                est_h = estimate_height_m(labels)

                f = QgsFeature()
                f.setGeometry(clipped)
                f.setAttributes([floor_text, kind, num, round(est_h, 2),
                                  res["confidence"], res["b5000_overlap"],
                                  res["emap_overlap"], res["sub_id"]])
                pr.addFeature(f)
                added_this += 1
                stats[res["confidence"]] = stats.get(res["confidence"], 0) + 1

        total_added += added_this
        print(f"    -> {added_this} sub-section polygons added")

    mem_layer.updateExtents()

    colours = {"high": QColor(0, 170, 0), "medium": QColor(230, 160, 0),
               "low": QColor(220, 60, 100), "unverified": QColor(120, 120, 120)}
    categories = []
    for cat_val, col in colours.items():
        sym = QgsSymbol.defaultSymbol(QgsWkbTypes.PolygonGeometry)
        sym.deleteSymbolLayer(0)
        fill_sl = QgsSimpleFillSymbolLayer()
        fill_sl.setBrushStyle(0)
        fill_sl.setStrokeColor(col)
        fill_sl.setStrokeWidth(0.5)
        sym.appendSymbolLayer(fill_sl)
        categories.append(QgsRendererCategory(cat_val, sym, cat_val))
    mem_layer.setRenderer(QgsCategorizedSymbolRenderer("confidence", categories))

    QgsProject.instance().addMapLayer(mem_layer)
    iface.mapCanvas().refresh()

    print("\n" + "=" * 70)
    print(f"[DONE] \"{OUT_LAYER_NAME}\" -- {total_added} sub-section polygons")
    print("       Attributes: floor_text | floor_kind | floor_number | "
          "est_height_m | confidence | b5000_overlap | emap_overlap | sub_id")
    print("-" * 70)
    print("Confidence breakdown:")
    for k in ("high", "medium", "low", "unverified"):
        n_ = stats.get(k, 0)
        pct = 100.0 * n_ / total_added if total_added else 0.0
        print(f"    {k:10s}: {n_:5d}  ({pct:5.1f}%)")
    ocr_rate = 100.0 * ocr_success / total_added if total_added else 0.0
    print(f"Floor-height OCR success: {ocr_success}/{total_added} ({ocr_rate:.1f}%)")
    if not _OCR_AVAILABLE:
        print("  OCR disabled entirely -- see DIAGNOSTIC above.")
    elif all_diag:
        from collections import Counter
        reasons = Counter(all_diag)
        print("  Failure reasons:")
        for reason, cnt in reasons.most_common(6):
            print(f"    {reason}: {cnt}")
        print("  If 'no_glyph_clusters_in_subface' dominates: the sub-face split "
              "may be excluding the region containing the label text -- try "
              "lowering HATCH_STRIP_KERNEL_PX to 1, or set DEBUG_SAVE_CROPS=True "
              f"and inspect crops written to {_DEBUG_DIR}")
    print("=" * 70)


if __name__ == "__main__" or True:
    main()