#!/usr/bin/env python3
"""
gen_hls_data.py — Generate HLS test data for vj_evaluate_window_fixed.

Outputs: hls/hls_test_data.h  +  console analysis report.

Sections:
  1. Small crop II/SII as C arrays (target <= 64x64)
  2. Two golden test vectors (pass + reject), float == fixed confirmed
  3. Single-scale vj_scaled_feature_t table for the pass window size
  4. Full fixed-point cascade tables (stages / wcs / features_fixed)
  5. __int128 → ap_int<N> bit-width analysis (theoretical + empirical)
"""
import xml.etree.ElementTree as ET
import numpy as np
import cv2, os, sys, math

# ──────────────────────────────────────────────────────────
# Cascade parsing (mirrors verify scripts)
# ──────────────────────────────────────────────────────────
CASCADE_XML = "haarcascade_frontalface_default.xml"
WIN_W = WIN_H = 24
SCALE_FACTOR  = 1.2
Q_THRESH, Q_LRVAL = 15, 10
WCODE_NEG1, WCODE_POS2, WCODE_POS3 = 0, 1, 2

tree = ET.parse(CASCADE_XML)
cn   = tree.getroot().find(".//cascade")

def weight_to_code(w):
    return WCODE_NEG1 if w < 0 else (WCODE_POS2 if w < 2.5 else WCODE_POS3)

def apply_weight(rs, code):
    return -rs if code == WCODE_NEG1 else (rs<<1 if code == WCODE_POS2 else (rs<<1)+rs)

def to_q(v, bits):
    s = v * (1 << bits)
    return int(s + 0.5 if s >= 0 else s - 0.5)

# Float features
features_float = []
for feat in cn.find("features").findall("_"):
    rects = []
    for r in feat.find("rects").findall("_"):
        p = r.text.strip().split()
        rects.append((int(p[0]),int(p[1]),int(p[2]),int(p[3]),float(p[4])))
    features_float.append(rects)
NUM_FEATURES = len(features_float)

# Fixed-code features: (x,y,w,h,code)
features_code = [[(x,y,w,h,weight_to_code(wt)) for x,y,w,h,wt in rects]
                 for rects in features_float]

# Cascade tables (Python mirrors of vj_cascade_fixed_build)
stages_raw = []    # (st_f, [(fi, th_f, lv_f, rv_f)])
stages_fixed = []  # (st_q10, [(fi, th_q15, lv_q10, rv_q10)])
for stage in cn.find("stages").findall("_"):
    st_f = float(stage.find("stageThreshold").text.strip())
    wcs_f, wcs_q = [], []
    for wc in stage.find("weakClassifiers").findall("_"):
        internals = wc.find("internalNodes").text.strip().split()
        leafs     = wc.find("leafValues").text.strip().split()
        fi = int(internals[2]); th_f = float(internals[3])
        lv_f = float(leafs[0]); rv_f = float(leafs[1])
        wcs_f.append((fi, th_f, lv_f, rv_f))
        wcs_q.append((fi, to_q(th_f,Q_THRESH), to_q(lv_f,Q_LRVAL), to_q(rv_f,Q_LRVAL)))
    stages_raw.append((st_f, wcs_f))
    stages_fixed.append((to_q(st_f, Q_LRVAL), wcs_q))
NUM_STAGES = len(stages_fixed)

# Cumulative weak_start_idx
weak_all = []         # flat list of (fi, th_q15, lv_q10, rv_q10)
stage_meta = []       # (num_weak, weak_start_idx, stage_threshold_q10)
for (st_q10, wcs_q) in stages_fixed:
    stage_meta.append((len(wcs_q), len(weak_all), st_q10))
    weak_all.extend(wcs_q)
NUM_WEAK_TOTAL = len(weak_all)

# ──────────────────────────────────────────────────────────
# Integral image helpers
# ──────────────────────────────────────────────────────────
def build_ii(gray):
    g  = gray.astype(np.int64)
    ii  = np.zeros((gray.shape[0]+1, gray.shape[1]+1), dtype=np.int64)
    sii = np.zeros((gray.shape[0]+1, gray.shape[1]+1), dtype=np.int64)
    ii[1:,1:]  = np.cumsum(np.cumsum(g,    axis=0), axis=1)
    sii[1:,1:] = np.cumsum(np.cumsum(g**2, axis=0), axis=1)
    return ii, sii

def rsum(ii, x, y, w, h):
    return int(ii[y+h,x+w] - ii[y,x+w] - ii[y+h,x] + ii[y,x])

# ──────────────────────────────────────────────────────────
# Evaluators
# ──────────────────────────────────────────────────────────
def eval_float(ii, sii, wx, wy, ww, wh):
    inv = 1.0/(ww*wh)
    s  = rsum(ii,  wx,wy,ww,wh)
    sq = rsum(sii, wx,wy,ww,wh)
    mean = s*inv; var = sq*inv - mean*mean
    stdev = var**0.5 if var > 0 else 1.0
    sx=ww/WIN_W; sy=wh/WIN_H
    for (st_f, wcs_f) in stages_raw:
        ss = 0.0
        for (fi,th_f,lv_f,rv_f) in wcs_f:
            fv = sum(rsum(ii,wx+int(rx*sx),wy+int(ry*sy),max(1,int(rw*sx)),max(1,int(rh*sy)))*wt
                    for rx,ry,rw,rh,wt in features_float[fi]) * inv
            ss += lv_f if fv < th_f*stdev else rv_f
        if ss < st_f: return 0
    return 1

def _cmp4(L, T, va_sq, area):
    if va_sq <= 0: return L < T*area
    if T > 0:      return (L < 0) or  (L*L < T*T*va_sq)
    if T < 0:      return (L < 0) and (L*L > T*T*va_sq)
    return L < 0

def build_scaled_feats_table(ww, wh):
    sx=ww/WIN_W; sy=wh/WIN_H
    return [[(int(x*sx),int(y*sy),max(1,int(w*sx)),max(1,int(h*sy)),code)
             for x,y,w,h,code in rects]
            for rects in features_code]

def eval_fixed(ii, sii, wx, wy, ww, wh, sf):
    area = ww*wh; s=rsum(ii,wx,wy,ww,wh); sq=rsum(sii,wx,wy,ww,wh)
    va_sq = sq*area - s*s; scale_q = 1<<Q_THRESH
    for (st_q10, wcs_q) in stages_fixed:
        ss = 0
        for (fi,th_q15,lv_q10,rv_q10) in wcs_q:
            fv = sum(apply_weight(rsum(ii,wx+dx,wy+dy,rw,rh),code)
                    for dx,dy,rw,rh,code in sf[fi])
            ss += lv_q10 if _cmp4(fv*scale_q,th_q15,va_sq,area) else rv_q10
        if ss < st_q10: return 0
    return 1

# ──────────────────────────────────────────────────────────
# SECTION 1+2: Find pass/reject windows in testt.jpg
# ──────────────────────────────────────────────────────────
img  = cv2.imread("test/testt.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
IMG_H, IMG_W = gray.shape
ii_full, sii_full = build_ii(gray)

print("Scanning testt.jpg for test vectors...")
pass_windows   = []
scale = 1.0
while True:
    ww = int(WIN_W*scale+0.5); wh = int(WIN_H*scale+0.5)
    if ww > IMG_W or wh > IMG_H: break
    ystep = 1 if scale > 2.0 else 2
    step  = max(1, int(ystep*scale+0.5))
    sf    = build_scaled_feats_table(ww, wh)
    for y in range(0, IMG_H-wh+1, step):
        for x in range(0, IMG_W-ww+1, step):
            if eval_fixed(ii_full,sii_full,x,y,ww,wh,sf):
                rf = eval_float(ii_full,sii_full,x,y,ww,wh)
                if rf == 1:
                    pass_windows.append((x,y,ww,wh))
    scale *= SCALE_FACTOR

assert pass_windows, "No pass windows found in testt.jpg!"

# Pick smallest window for minimum crop
pass_windows.sort(key=lambda t: t[2])
px, py, pw, ph = pass_windows[0]
print(f"  First pass window: ({px},{py},{pw},{ph})  count={len(pass_windows)}")

# Build crop: add margin so crop fits within 64x64 if possible
margin = min(6, (64-pw)//2, (64-ph)//2)
margin = max(2, margin)
cx = max(0, px-margin); cy = max(0, py-margin)
cw = min(IMG_W-cx, pw+2*margin); ch = min(IMG_H-cy, ph+2*margin)
# Clamp to 64x64
if cw > 64: cw = 64
if ch > 64: ch = 64
# Ensure pass window fits in crop
assert px-cx+pw <= cw and py-cy+ph <= ch, "Pass window doesn't fit in crop!"
crop = gray[cy:cy+ch, cx:cx+cw].copy()
ii_crop, sii_crop = build_ii(crop)
ii_stride = cw + 1

# Pass window in crop coords
pass_cx = px - cx; pass_cy = py - cy
assert eval_float(ii_crop,sii_crop,pass_cx,pass_cy,pw,ph) == 1, "float fail on crop!"
assert eval_fixed(ii_crop,sii_crop,pass_cx,pass_cy,pw,ph,
                  build_scaled_feats_table(pw,ph)) == 1, "fixed fail on crop!"
print(f"  Crop: origin=({cx},{cy}) size={cw}x{ch}")
print(f"  Pass window in crop: ({pass_cx},{pass_cy},{pw},{ph})  CONFIRMED float=1 fixed=1")

# Find reject window at same scale within crop
sf_crop = build_scaled_feats_table(pw, ph)
reject_window = None
for ry2 in range(0, ch-ph+1, 1):
    for rx2 in range(0, cw-pw+1, 1):
        if rx2==pass_cx and ry2==pass_cy: continue
        if (eval_float(ii_crop,sii_crop,rx2,ry2,pw,ph) == 0 and
            eval_fixed(ii_crop,sii_crop,rx2,ry2,pw,ph,sf_crop) == 0):
            reject_window = (rx2, ry2, pw, ph)
            break
    if reject_window: break

if not reject_window:
    # Fallback: use a different scale if available
    for (x2,y2,ww2,wh2) in [(0,0,min(cw,24),min(ch,24))]:
        if (eval_float(ii_crop,sii_crop,x2,y2,ww2,wh2)==0 and
            eval_fixed(ii_crop,sii_crop,x2,y2,ww2,wh2,build_scaled_feats_table(ww2,wh2))==0):
            reject_window = (x2,y2,ww2,wh2)
            break

assert reject_window, "Could not find a reject window in crop!"
rx, ry, rw, rh = reject_window
print(f"  Reject window in crop: ({rx},{ry},{rw},{rh})  CONFIRMED float=0 fixed=0")

# ──────────────────────────────────────────────────────────
# SECTION 3: Scaled features table for pass window size
# ──────────────────────────────────────────────────────────
sf_table = build_scaled_feats_table(pw, ph)
sf_bytes = NUM_FEATURES * 3 * 4 * 2   # 3 rects * 4 int16_t * 2 bytes
print(f"\nScaled features ({pw}x{ph}): {NUM_FEATURES} features, {sf_bytes:,} bytes ({sf_bytes/1024:.1f} KB)")

# ──────────────────────────────────────────────────────────
# SECTION 5: Bit-width analysis
# ──────────────────────────────────────────────────────────
print("\nBit-width analysis...")

# Theoretical (from QUANT_ANALYSIS.md)
fvr_max_theory   = 15_343_754          # feat_val_raw max absolute
L_max_theory     = fvr_max_theory * (1 << Q_THRESH)
L_sq_theory      = L_max_theory ** 2

th_q15_max       = round(0.6606 * (1 << Q_THRESH))
th_q15_sq_theory = th_q15_max ** 2
area_max         = 588289
variance_max     = 8857
va_sq_max_theory = variance_max * area_max**2
R_sq_theory      = th_q15_sq_theory * va_sq_max_theory

combined_theory  = max(L_sq_theory, R_sq_theory)
bits_unsigned_theory = math.ceil(math.log2(combined_theory + 1))
bits_signed_theory   = bits_unsigned_theory + 1  # +1 for sign bit

# Empirical: scan testt.jpg, track max |L| and T^2*va_sq per classifier evaluated
L_abs_max_emp    = 0
R_sq_max_emp     = 0
scale = 1.0
while True:
    ww = int(WIN_W*scale+0.5); wh = int(WIN_H*scale+0.5)
    if ww > IMG_W or wh > IMG_H: break
    ystep = 1 if scale > 2.0 else 2
    step  = max(1, int(ystep*scale+0.5))
    sf2   = build_scaled_feats_table(ww, wh)
    area2 = ww * wh
    for y in range(0, IMG_H-wh+1, step):
        for x in range(0, IMG_W-ww+1, step):
            s2  = rsum(ii_full, x,y,ww,wh)
            sq2 = rsum(sii_full,x,y,ww,wh)
            va_sq2 = sq2*area2 - s2*s2
            scale_q = 1 << Q_THRESH
            for si,(st_q10,wcs_q) in enumerate(stages_fixed):
                ss = 0
                reject = False
                for (fi,th_q15,lv_q10,rv_q10) in wcs_q:
                    fv = sum(apply_weight(rsum(ii_full,x+dx,y+dy,rw2,rh2),code)
                            for dx,dy,rw2,rh2,code in sf2[fi])
                    L_val = fv * scale_q
                    L_abs_max_emp = max(L_abs_max_emp, abs(L_val))
                    R_sq_val = th_q15*th_q15*va_sq2
                    R_sq_max_emp = max(R_sq_max_emp, R_sq_val)
                    ss += lv_q10 if _cmp4(L_val,th_q15,va_sq2,area2) else rv_q10
                if ss < st_q10:
                    break
    scale *= SCALE_FACTOR

L_sq_emp = L_abs_max_emp ** 2
combined_emp  = max(L_sq_emp, R_sq_max_emp)
bits_unsigned_emp = math.ceil(math.log2(combined_emp + 1)) if combined_emp > 0 else 1
bits_signed_emp   = bits_unsigned_emp + 1

print(f"  Theoretical: L_sq_max={L_sq_theory:.2e}  R_sq_max={R_sq_theory:.2e}")
print(f"               need ap_uint<{bits_unsigned_theory}>  or ap_int<{bits_signed_theory}>")
print(f"  Empirical (testt.jpg scan):")
print(f"    |L|_max = {L_abs_max_emp:,}  L^2_max = {L_sq_emp:.2e}")
print(f"    T^2*va_sq_max = {R_sq_max_emp:.2e}")
print(f"    combined_max  = {combined_emp:.2e}")
print(f"    need ap_uint<{bits_unsigned_emp}>  or ap_int<{bits_signed_emp}>")

# ──────────────────────────────────────────────────────────
# C array formatter helpers
# ──────────────────────────────────────────────────────────
def fmt_array_u32(arr, cols=8):
    lines = ["    "]
    for i, v in enumerate(arr):
        lines[-1] += f"{v}u"
        if i < len(arr)-1:
            lines[-1] += ","
            if (i+1) % cols == 0:
                lines.append("    ")
            else:
                lines[-1] += " "
    return "\n".join(lines)

def fmt_array_u64(arr, cols=6):
    lines = ["    "]
    for i, v in enumerate(arr):
        lines[-1] += f"{v}ULL"
        if i < len(arr)-1:
            lines[-1] += ","
            if (i+1) % cols == 0:
                lines.append("    ")
            else:
                lines[-1] += " "
    return "\n".join(lines)

# ──────────────────────────────────────────────────────────
# Write hls_test_data.h
# ──────────────────────────────────────────────────────────
os.makedirs("hls", exist_ok=True)
out_path = "hls/hls_test_data.h"

ii_flat  = ii_crop.flatten().tolist()    # (ch+1)*(cw+1) int64 → output as uint32
sii_flat = sii_crop.flatten().tolist()

with open(out_path, "w", encoding="utf-8") as f:
    w = f.write

    w("/* hls_test_data.h — auto-generated by experiments/gen_hls_data.py\n")
    w(" * DO NOT EDIT: regenerate with: python experiments/gen_hls_data.py\n")
    w(" *\n")
    w(f" * Source image : test/testt.jpg\n")
    w(f" * Crop origin  : ({cx}, {cy})   size: {cw} x {ch}\n")
    w(f" * Pass window  : ({pass_cx},{pass_cy},{pw},{ph})  (in crop coords)\n")
    w(f" * Reject window: ({rx},{ry},{rw},{rh})  (in crop coords, same scale)\n")
    w(f" * ap_int<N>    : N = {bits_signed_theory} (theoretical)  "
      f"/ N = {bits_signed_emp} (empirical testt.jpg)\n")
    w(" */\n\n")
    w("#pragma once\n")
    w("#include <stdint.h>\n")
    w('#include "vj_fixed.h"\n\n')

    # Section 1: II / SII
    w("/* =========================================================\n")
    w(" * Section 1: Test integral images\n")
    w(f" * Crop [{cy}:{cy+ch}, {cx}:{cx+cw}] of testt.jpg\n")
    w(" * ========================================================= */\n")
    w(f"#define HLS_TEST_CROP_X      {cx}\n")
    w(f"#define HLS_TEST_CROP_Y      {cy}\n")
    w(f"#define HLS_TEST_CROP_W      {cw}\n")
    w(f"#define HLS_TEST_CROP_H      {ch}\n")
    w(f"#define HLS_TEST_II_STRIDE   {cw+1}\n")
    w(f"#define HLS_TEST_II_LEN      {(cw+1)*(ch+1)}\n\n")
    w(f"static const uint32_t g_test_ii[HLS_TEST_II_LEN] = {{\n")
    w(fmt_array_u32(ii_flat, 8))
    w("\n};\n\n")
    w(f"static const uint64_t g_test_sii[HLS_TEST_II_LEN] = {{\n")
    w(fmt_array_u64(sii_flat, 6))
    w("\n};\n\n")

    # Section 2: Test vectors
    w("/* =========================================================\n")
    w(" * Section 2: Golden test vectors (float == fixed confirmed)\n")
    w(" * ========================================================= */\n")
    w(f"#define HLS_PASS_X       {pass_cx}\n")
    w(f"#define HLS_PASS_Y       {pass_cy}\n")
    w(f"#define HLS_PASS_W       {pw}\n")
    w(f"#define HLS_PASS_H       {ph}\n")
    w(f"#define HLS_PASS_EXPECTED  1\n\n")
    w(f"#define HLS_REJECT_X     {rx}\n")
    w(f"#define HLS_REJECT_Y     {ry}\n")
    w(f"#define HLS_REJECT_W     {rw}\n")
    w(f"#define HLS_REJECT_H     {rh}\n")
    w(f"#define HLS_REJECT_EXPECTED  0\n\n")

    # Section 3: Scaled features
    w("/* =========================================================\n")
    w(f" * Section 3: Precomputed scaled features for win={pw}x{ph}\n")
    w(f" * {NUM_FEATURES} features x 3 rects x 4 int16  = {sf_bytes:,} bytes ({sf_bytes/1024:.1f} KB)\n")
    w(" * ========================================================= */\n")
    w(f"#define HLS_SCALE_WIN_W  {pw}\n")
    w(f"#define HLS_SCALE_WIN_H  {ph}\n")
    w(f"#define HLS_NUM_FEATURES {NUM_FEATURES}\n\n")
    w(f"static const vj_scaled_feature_t g_scaled_feats[HLS_NUM_FEATURES] = {{\n")
    parts = []
    for fi, rects in enumerate(sf_table):
        # rects is list of (dx,dy,rw,rh,code) — vj_scaled_feature_t has only dx,dy,rw,rh
        rr = []
        for (dx,dy,rw2,rh2,code) in rects:
            rr.append(f"{{{dx},{dy},{rw2},{rh2}}}")
        while len(rr) < 3:
            rr.append("{0,0,0,0}")
        parts.append("{{" + ",".join(rr) + "}}")   # struct{array{rect,rect,rect}}
    # output 4 per line
    lines = ["    "]
    for i, p in enumerate(parts):
        lines[-1] += p
        if i < len(parts)-1:
            lines[-1] += ","
            if (i+1) % 4 == 0:
                lines.append("    ")
            else:
                lines[-1] += " "
    w("\n".join(lines))
    w("\n};\n\n")

    # Section 4: Cascade tables
    w("/* =========================================================\n")
    w(f" * Section 4: Fixed-point cascade tables\n")
    w(f" * stages: {NUM_STAGES}  wcs: {NUM_WEAK_TOTAL}  features: {NUM_FEATURES}\n")
    w(" * ========================================================= */\n")
    w(f"#define HLS_NUM_STAGES      {NUM_STAGES}\n")
    w(f"#define HLS_NUM_WEAK_TOTAL  {NUM_WEAK_TOTAL}\n\n")

    # stages
    w(f"static const vj_stage_fixed_t g_stages[HLS_NUM_STAGES] = {{\n")
    stage_parts = []
    for (nw, ws, st_q10) in stage_meta:
        stage_parts.append(f"{{{nw},{ws},{st_q10}}}")
    w("    " + ",\n    ".join(stage_parts))
    w("\n};\n\n")

    # weak classifiers
    w(f"static const vj_wc_fixed_t g_weak_classifiers[HLS_NUM_WEAK_TOTAL] = {{\n")
    wc_parts = []
    for (fi2, th_q15, lv_q10, rv_q10) in weak_all:
        wc_parts.append(f"{{{fi2},{th_q15},{lv_q10},{rv_q10}}}")
    lines = ["    "]
    for i, p in enumerate(wc_parts):
        lines[-1] += p
        if i < len(wc_parts)-1:
            lines[-1] += ","
            if (i+1) % 4 == 0:
                lines.append("    ")
            else:
                lines[-1] += " "
    w("\n".join(lines))
    w("\n};\n\n")

    # features_fixed
    w(f"static const vj_feature_fixed_t g_features_fixed[HLS_NUM_FEATURES] = {{\n")
    feat_parts = []
    for fi2, flt_rects in enumerate(features_float):
        rr = []
        for (x2,y2,w2,h2,wt) in flt_rects:
            code = weight_to_code(wt)
            rr.append(f"{{{x2},{y2},{w2},{h2},{code}}}")
        num_r = len(flt_rects)
        while len(rr) < 3:
            rr.append("{0,0,0,0,0}")
        feat_parts.append("{{{{{}}},{}}}" .format(",".join(rr), num_r))
    lines = ["    "]
    for i, p in enumerate(feat_parts):
        lines[-1] += p
        if i < len(feat_parts)-1:
            lines[-1] += ","
            if (i+1) % 3 == 0:
                lines.append("    ")
            else:
                lines[-1] += " "
    w("\n".join(lines))
    w("\n};\n\n")

    # Section 5: ap_int analysis as comments
    w("/* =========================================================\n")
    w(" * Section 5: ap_int<N> bit-width memo\n")
    w(f" *\n")
    w(f" * cmp_lhs_lt_rhs() compares L^2  vs  T^2 * va_sq\n")
    w(f" *   L       = feat_val_raw << {Q_THRESH}          (int64)\n")
    w(f" *   T       = threshold_q15                  (int32)\n")
    w(f" *   va_sq   = sqsum*area - sum^2             (int64, always >= 0)\n")
    w(f" *\n")
    w(f" * Theoretical bounds (QUANT_ANALYSIS.md):\n")
    w(f" *   feat_val_raw_max = {fvr_max_theory:,}\n")
    w(f" *   L_max            = {L_max_theory:.3e}  "
      f"L^2_max = {L_sq_theory:.3e}\n")
    w(f" *   T_max            = {th_q15_max}  "
      f"va_sq_max = {va_sq_max_theory:.3e}\n")
    w(f" *   T^2*va_sq_max    = {R_sq_theory:.3e}\n")
    w(f" *   combined_max     = {combined_theory:.3e}\n")
    w(f" *   -> ap_uint<{bits_unsigned_theory}>  /  ap_int<{bits_signed_theory}> (signed)\n")
    w(f" *\n")
    w(f" * Empirical (testt.jpg full scan):\n")
    w(f" *   |L|_max          = {L_abs_max_emp:,}\n")
    w(f" *   L^2_max          = {L_sq_emp:.3e}\n")
    w(f" *   T^2*va_sq_max    = {R_sq_max_emp:.3e}\n")
    w(f" *   combined_max     = {combined_emp:.3e}\n")
    w(f" *   -> ap_uint<{bits_unsigned_emp}>  /  ap_int<{bits_signed_emp}> (signed)\n")
    w(f" *\n")
    w(f" * Recommended: ap_uint<{bits_unsigned_theory}> (use theoretical, conservative).\n")
    w(f" *   HLS pragma: ap_uint<{bits_unsigned_theory}> L_sq, R_sq;\n")
    w(f" * ========================================================= */\n")
    w(f"#define HLS_APINT_BITS_LSQR     {math.ceil(math.log2(L_sq_theory+1))}\n")
    w(f"#define HLS_APINT_BITS_RSQR     {math.ceil(math.log2(R_sq_theory+1))}\n")
    w(f"#define HLS_APINT_N_UNSIGNED    {bits_unsigned_theory}\n")
    w(f"#define HLS_APINT_N_SIGNED      {bits_signed_theory}\n")

    # Section 6: convenience aliases for vj_cascade_top.cpp
    w("\n/* =========================================================\n")
    w(" * Section 6: convenience aliases for vj_cascade_top.cpp\n")
    w(" * ========================================================= */\n\n")
    w("/* Stride alias */\n")
    w(f"#define TEST_II_STRIDE  HLS_TEST_II_STRIDE   /* = {cw+1} */\n\n")
    w("/* Array alias: vj_cascade_top.cpp references g_scaled */\n")
    w("#define g_scaled  g_scaled_feats\n\n")
    w("/* Assembled cascade struct — field order matches vj_cascade_fixed_t */\n")
    w("static const vj_cascade_fixed_t g_cascade = {\n")
    w(f"    {NUM_STAGES},                /* num_stages       */\n")
    w(f"    {NUM_WEAK_TOTAL},            /* num_weak_total   */\n")
    w(f"    {NUM_FEATURES},              /* num_features     */\n")
    w(f"    {WIN_W}, {WIN_H},            /* window_w, window_h (base window) */\n")
    w("    g_stages,\n")
    w("    g_weak_classifiers,\n")
    w("    g_features_fixed\n")
    w("};\n")

print(f"\nOutput: {out_path}")
fsize = os.path.getsize(out_path)
print(f"File size: {fsize:,} bytes ({fsize/1024:.1f} KB)")
print("\nDone.")
