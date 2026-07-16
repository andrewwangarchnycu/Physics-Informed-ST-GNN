# -*- coding: utf-8 -*-
"""
nlsc_building_vectorize.py  [v3]
=================================
QGIS Python Script - NLSC WMTS TOPO 1:1000 Building Vectorizer
High-precision mode: resolves individual building height sections
(roof protrusions, volume steps, entrance canopies).

Usage:
    1. In the QGIS Layers Panel, click the polygon layer that defines the area
       to extract buildings from.  Must be a vector Polygon layer (not WMTS).
    2. Plugins > Python Console > Show Editor > Run.

Key algorithm v3:
    - Zoom 18 default (0.6 m/px) -- fine enough to separate height-section walls
    - Processes each extent-layer feature independently (keeps tile count low
      even at high zoom, auto-adapts zoom per-feature by physical size)
    - NO binary_closing on the pink mask -- closing was filling the 1-2 px
      black interior walls that divide height sections, merging them into one blob
    - Direct connected-component labeling of raw pink pixels: each component is
      naturally isolated by the black boundary/interior lines in the NLSC tile
    - Sub-pixel contour (cv2 CHAIN_APPROX_TC89_KCOS + approxPolyDP) or skimage
      find_contours fallback
    - OCR at pink-region centroid (floor label always centred inside its section)
    - Physical area filter (MIN_AREA_M2) -- strips OCR text blobs and 1-px noise

Attributes: floor_text (e.g. "3R", "T"), floors (int; T->1)
"""

import sys, os, io, re, warnings
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---- User-configurable -------------------------------------------------------
ZOOM           = 18     # default tile zoom; 18=0.60m/px, 19=0.30m/px
ZOOM_MAX       = 19     # cap for auto-zoom-up on small features
BUFFER_TILES   = 1      # extra tile margin on each side
MIN_AREA_M2    = 2.0    # drop polygons smaller than this physical area (m2)
SIMPLIFY_M     = 0.3    # Douglas-Peucker tolerance in metres (not pixels)
FORCE_LAYER_ID = ""     # "" = auto-detect from WMTS capabilities
OUT_LAYER_NAME = "buildings_nlsc"
FLOOR_OCR      = True

WMTS_ROOT     = "https://wmts.nlsc.gov.tw/wmts"
WMTS_CAPS_URL = ("https://maps.nlsc.gov.tw/wmtsTOPO01K/"
                 "?SERVICE=WMTS&REQUEST=GetCapabilities")
TILE_URL_TMPL = WMTS_ROOT + "/{layer}/default/GoogleMapsCompatible/{z}/{y}/{x}"

BLDG_R, BLDG_G, BLDG_B = 255, 117, 117
MAX_TILES = 100

_SCRIPT_DIR = (os.path.dirname(os.path.abspath(__file__))
               if "__file__" in dir() else os.getcwd())

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
    print("[OK] OpenCV (cv2) -- TC89 contour + approxPolyDP enabled")
except ImportError:
    print("[INFO] cv2 not available -- trying skimage")

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
    if not _CV2:
        print("[OK] scikit-image loaded (fallback contour)")
except ImportError:
    if not _CV2:
        print("[WARN] No cv2 or skimage -- boundary will be basic")

_OCR_AVAILABLE = False
_tess = None
try:
    import pytesseract as _tess
    for _tp in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
        if os.path.exists(_tp):
            _tess.pytesseract.tesseract_cmd = _tp
            break
    _OCR_AVAILABLE = True
    print("[OK] pytesseract -- floor OCR enabled")
except ImportError:
    print("[WARN] pytesseract not found -- floor labels will be empty")

from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry,
    QgsPointXY, QgsField, QgsFields, QgsCoordinateReferenceSystem,
    QgsCoordinateTransform, QgsRectangle, QgsWkbTypes,
    QgsSymbol, QgsSimpleFillSymbolLayer, QgsSingleSymbolRenderer,
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
_detected_layer_id = [None]   # cache

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
    """Return (zoom, tx0, ty0, tx1, ty1) fitting within MAX_TILES."""
    z = desired_zoom
    while z > 14:
        tx0, ty0 = _3857_to_tile(ext_3857.xMinimum(), ext_3857.yMaximum(), z)
        tx1, ty1 = _3857_to_tile(ext_3857.xMaximum(), ext_3857.yMinimum(), z)
        tx0 -= buf; ty0 -= buf; tx1 += buf; ty1 += buf
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= MAX_TILES:
            return z, tx0, ty0, tx1, ty1
        z -= 1
    return z, tx0, ty0, tx1, ty1


def fetch_tile_mosaic(ext_3857, layer_id, desired_zoom=ZOOM, buf=BUFFER_TILES):
    """
    Fetch WMTS tiles for ext_3857.
    Returns (mosaic_rgba, mosaic_xmin, mosaic_ymax, pixel_size_m).
    """
    z, tx0, ty0, tx1, ty1 = _choose_zoom(ext_3857, desired_zoom, buf)
    ncols = tx1 - tx0 + 1
    nrows = ty1 - ty0 + 1
    print(f"    fetch {ncols}x{nrows} tiles @ z={z} "
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


# =============================================================================
# 4. Building mask -- NO binary_closing
# =============================================================================
def extract_building_mask(mosaic_rgba):
    """
    Boolean mask: True where NLSC building pink (255,117,117) pixel exists.

    IMPORTANT: we deliberately do NOT apply binary_closing here.
    The NLSC tiles use 1-3 px black lines to separate adjacent building
    height sections (roof steps, volume changes, canopy overhangs).
    binary_closing would fill those lines and merge the sections into one
    polygon -- which is exactly what v1/v2 suffered from.
    """
    r, g, b, a = (mosaic_rgba[:, :, i] for i in range(4))
    return (r == BLDG_R) & (g == BLDG_G) & (b == BLDG_B) & (a > 200)


# =============================================================================
# 5. Sub-pixel contour extraction
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


def extract_building_polygons(mask, mosaic_xmin, mosaic_ymax, px_m):
    """
    Label connected pink components and trace each one's outer contour.

    Because we removed binary_closing, each connected component is already
    one naturally-separated height section (the black interior lines in
    the NLSC tile act as barriers between sections).

    Returns list of (QgsGeometry_EPSG3857, (cx_px, cy_px)).
    """
    # 4-connectivity: diagonal connections don't merge sections across an
    # angled black separation line
    labeled, n = ndi.label(mask)

    min_area_px = max(4, int(MIN_AREA_M2 / (px_m ** 2)))
    tol_px = max(1.0, SIMPLIFY_M / px_m)

    results = []
    for lbl in range(1, n + 1):
        comp = (labeled == lbl)
        npx  = int(comp.sum())
        if npx < min_area_px:
            continue

        sl = ndi.find_objects(comp)[0]
        r0 = max(sl[0].start - 1, 0)
        r1 = min(sl[0].stop  + 1, mask.shape[0])
        c0 = max(sl[1].start - 1, 0)
        c1 = min(sl[1].stop  + 1, mask.shape[1])
        patch = comp[r0:r1, c0:c1]

        if _CV2:
            rc = _contour_cv2((patch.astype(np.uint8)) * 255, tol_px)
        elif _SKIMAGE:
            rc = _contour_skimage(patch, tol_px)
        else:
            rc = _contour_fallback(patch)

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

        # Centroid of the pink blob (not contour centroid) for reliable OCR
        cy_px = int((sl[0].start + sl[0].stop) / 2)
        cx_px = int((sl[1].start + sl[1].stop) / 2)
        results.append((geom, (cx_px, cy_px), npx))

    return results


# =============================================================================
# 6. Floor OCR at centroid
# =============================================================================
_FLOOR_RE = re.compile(r"\b(\d{1,2}[Rr]|[Tt])\b")

def _parse_floor(text):
    m = _FLOOR_RE.search(text)
    if not m:
        return "", 0
    raw = m.group(1).upper()
    return ("T", 1) if raw == "T" else (raw, int(raw[:-1]))


def ocr_floor_centroid(mosaic_rgba, cx_px, cy_px, px_m):
    """
    OCR around the pink-region centroid.
    Radius adapts to pixel size so physical coverage is ~25m regardless of zoom.
    Floor numbers are guaranteed centred inside their building section.
    """
    if not (_OCR_AVAILABLE and FLOOR_OCR):
        return "", 0

    radius = max(30, int(25 / px_m))   # ~25 m radius in pixels
    H, W = mosaic_rgba.shape[:2]
    x0, x1 = max(0, cx_px - radius), min(W, cx_px + radius)
    y0, y1 = max(0, cy_px - radius), min(H, cy_px + radius)
    crop = mosaic_rgba[y0:y1, x0:x1, :3]
    if crop.size == 0:
        return "", 0

    img = Image.fromarray(crop)
    # Upscale to ~200px height for OCR
    scale = max(1, 200 // max(img.height, 1))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    arr = np.array(img)
    is_text = (arr[:, :, 0] < 60) & (arr[:, :, 1] < 60) & (arr[:, :, 2] < 60)
    bw = np.where(is_text[:, :, None], 0, 255).astype(np.uint8)
    bw_img = Image.fromarray(np.repeat(bw, 3, axis=2))

    try:
        raw = _tess.image_to_string(
            bw_img,
            config="--psm 6 -c tessedit_char_whitelist=0123456789RrTt"
        ).strip()
        return _parse_floor(raw)
    except Exception:
        return "", 0


# =============================================================================
# 7. Main
# =============================================================================
def main():
    print("=" * 65)
    print("NLSC TOPO1K Building Vectorizer  [v3 -- high-precision mode]")
    print("=" * 65)

    # -- A. Extent layer -------------------------------------------------------
    extent_layer = iface.activeLayer()
    if extent_layer is None:
        print("[ERROR] No layer selected.\n"
              "        Click a polygon layer in the Layers Panel, then re-run.")
        return
    if not hasattr(extent_layer, "getFeatures"):
        print("[ERROR] Selected layer is not a vector layer.\n"
              "        Select a polygon vector layer (not a raster/WMTS).")
        return

    src_crs  = extent_layer.crs()
    crs_3857 = QgsCoordinateReferenceSystem("EPSG:3857")
    tr_fwd   = QgsCoordinateTransform(src_crs, crs_3857, QgsProject.instance())

    # Collect extent features
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

    # Global bounding box for WMTS layer auto-detect
    union_g = feats[0]
    for g in feats[1:]:
        union_g = union_g.combine(g)
    global_ext = union_g.boundingBox()

    print(f"[INFO] Extent layer : {extent_layer.name()}  "
          f"({len(feats)} feature(s))  CRS: {src_crs.authid()}")

    layer_id = _find_layer_id(global_ext, FORCE_LAYER_ID)

    # -- B. Output memory layer ------------------------------------------------
    mem_layer = QgsVectorLayer("Polygon?crs=EPSG:3857", OUT_LAYER_NAME, "memory")
    pr = mem_layer.dataProvider()
    fields = QgsFields()
    fields.append(QgsField("floor_text", QVariant.String))
    fields.append(QgsField("floors",     QVariant.Int))
    pr.addAttributes(fields)
    mem_layer.updateFields()

    total_added = 0

    # -- C. Process each extent feature independently --------------------------
    for fidx, clip_g in enumerate(feats, 1):
        feat_ext = clip_g.boundingBox()

        # Adaptive zoom: go higher for small features (single building),
        # auto-reduce for large features so tile count stays within MAX_TILES
        # Estimate feature diagonal in metres
        dx = feat_ext.xMaximum() - feat_ext.xMinimum()
        dy = feat_ext.yMaximum() - feat_ext.yMinimum()
        diag_m = (dx**2 + dy**2) ** 0.5
        if diag_m < 200:
            desired_z = min(ZOOM_MAX, ZOOM + 1)
        elif diag_m < 500:
            desired_z = ZOOM
        else:
            desired_z = max(14, ZOOM - 1)

        print(f"\n[Feature {fidx}/{len(feats)}] diag={diag_m:.0f}m, "
              f"target zoom={desired_z}")

        mosaic, xmin, ymax, px_m = fetch_tile_mosaic(
            feat_ext, layer_id, desired_z, BUFFER_TILES)
        print(f"    mosaic {mosaic.shape[1]}x{mosaic.shape[0]} px  "
              f"pixel={px_m:.4f} m  "
              f"min_area={max(4, int(MIN_AREA_M2/px_m**2))} px")

        mask = extract_building_mask(mosaic)
        bldgs = extract_building_polygons(mask, xmin, ymax, px_m)
        print(f"    raw components: {len(bldgs)}")

        added_this = 0
        for geom, (cx_px, cy_px), npx in bldgs:
            # Clip to this extent feature
            clipped = geom.intersection(clip_g)
            if clipped is None or clipped.isEmpty():
                continue
            floor_text, floors = ocr_floor_centroid(mosaic, cx_px, cy_px, px_m)
            f = QgsFeature()
            f.setGeometry(clipped)
            f.setAttributes([floor_text, floors])
            pr.addFeature(f)
            added_this += 1

        total_added += added_this
        print(f"    -> {added_this} building sections added")

    mem_layer.updateExtents()

    # -- D. Style: pink outline, no fill ---------------------------------------
    sym = QgsSymbol.defaultSymbol(QgsWkbTypes.PolygonGeometry)
    sym.deleteSymbolLayer(0)
    fill_sl = QgsSimpleFillSymbolLayer()
    fill_sl.setBrushStyle(0)
    fill_sl.setStrokeColor(QColor(220, 60, 100))
    fill_sl.setStrokeWidth(0.4)
    sym.appendSymbolLayer(fill_sl)
    mem_layer.setRenderer(QgsSingleSymbolRenderer(sym))

    QgsProject.instance().addMapLayer(mem_layer)
    iface.mapCanvas().refresh()

    print("\n" + "=" * 65)
    print(f"[DONE] \"{OUT_LAYER_NAME}\" -- {total_added} building sections")
    print("       Attributes: floor_text | floors")
    print("=" * 65)


if __name__ == "__main__" or True:
    main()