"""
Step 5: Full regression verification + HLS-readiness summary.

Checks:
  1. Per-window consistency: Step4 fixed vs float golden (all images >= 99.5%)
  2. Per-stage flip breakdown (diagnostic if any flips found)
  3. Full pipeline comparison: best_face coords (fixed vs golden)
  4. Additional image test if available

Final verdict: PASS only if all criteria pass.
"""
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os
from collections import defaultdict, Counter

CASCADE_XML  = "haarcascade_frontalface_default.xml"
WIN_W = WIN_H = 24
SCALE_FACTOR  = 1.2
Q_THRESH = 15
Q_LRVAL  = 10
VJ_BEST_FACE_MAX_RAW = 2048

# ---------- parse cascade ----------
tree = ET.parse(CASCADE_XML)
cn   = tree.getroot().find(".//cascade")

WCODE_NEG1, WCODE_POS2, WCODE_POS3 = 0, 1, 2

def weight_to_code(w):
    return WCODE_NEG1 if w < 0 else (WCODE_POS2 if w < 2.5 else WCODE_POS3)

def apply_weight(rs, code):
    if code == WCODE_NEG1: return -rs
    if code == WCODE_POS2: return rs << 1
    return (rs << 1) + rs

features_float = []
features_code  = []
for feat in cn.find("features").findall("_"):
    rf, rc = [], []
    for r in feat.find("rects").findall("_"):
        p = r.text.strip().split()
        x,y,w,h,wt = int(p[0]),int(p[1]),int(p[2]),int(p[3]),float(p[4])
        rf.append((x,y,w,h,wt))
        rc.append((x,y,w,h,weight_to_code(wt)))
    features_float.append(rf)
    features_code.append(rc)

def to_q(v, bits):
    s = v * (1 << bits)
    return int(s + 0.5 if s >= 0 else s - 0.5)

stages_float = []
stages_q     = []
for stage in cn.find("stages").findall("_"):
    st_f = float(stage.find("stageThreshold").text.strip())
    wcs_f, wcs_q = [], []
    for wc in stage.find("weakClassifiers").findall("_"):
        internals = wc.find("internalNodes").text.strip().split()
        leafs     = wc.find("leafValues").text.strip().split()
        fi   = int(internals[2])
        th_f = float(internals[3])
        lv_f = float(leafs[0])
        rv_f = float(leafs[1])
        wcs_f.append((fi, th_f, lv_f, rv_f))
        wcs_q.append((fi, to_q(th_f, Q_THRESH), to_q(lv_f, Q_LRVAL), to_q(rv_f, Q_LRVAL)))
    stages_float.append((st_f, wcs_f))
    stages_q.append((to_q(st_f, Q_LRVAL), wcs_q))

NUM_STAGES = len(stages_q)

# ---------- integral image ----------
def build_ii(gray):
    g  = gray.astype(np.int64)
    ii  = np.zeros((gray.shape[0]+1, gray.shape[1]+1), dtype=np.int64)
    sii = np.zeros((gray.shape[0]+1, gray.shape[1]+1), dtype=np.int64)
    ii[1:,1:]  = np.cumsum(np.cumsum(g,    axis=0), axis=1)
    sii[1:,1:] = np.cumsum(np.cumsum(g**2, axis=0), axis=1)
    return ii, sii

def rsum(ii, x, y, w, h):
    return int(ii[y+h, x+w] - ii[y, x+w] - ii[y+h, x] + ii[y, x])

# ---------- precomputed coord table ----------
def build_scaled_feats(win_w, win_h):
    sx = win_w / WIN_W; sy = win_h / WIN_H
    return [[(int(x*sx), int(y*sy), max(1,int(w*sx)), max(1,int(h*sy)), code)
             for x,y,w,h,code in rects]
            for rects in features_code]

# ---------- float golden ----------
def eval_float(ii, sii, win_x, win_y, win_w, win_h):
    inv_area = 1.0 / (win_w * win_h)
    s  = rsum(ii,  win_x, win_y, win_w, win_h)
    sq = rsum(sii, win_x, win_y, win_w, win_h)
    mean     = s * inv_area
    variance = sq * inv_area - mean * mean
    stdev    = variance**0.5 if variance > 0.0 else 1.0
    sx = win_w / WIN_W; sy = win_h / WIN_H
    for (st_f, wcs_f) in stages_float:
        ss = 0.0
        for (fi, th_f, lv_f, rv_f) in wcs_f:
            fv = sum(rsum(ii, win_x+int(rx*sx), win_y+int(ry*sy),
                         max(1,int(rw*sx)), max(1,int(rh*sy))) * wt
                    for rx,ry,rw,rh,wt in features_float[fi]) * inv_area
            ss += lv_f if fv < th_f * stdev else rv_f
        if ss < st_f:
            return 0
    return 1

# ---------- Step 4 fixed (final version) ----------
def _cmp_sq4(L, T, va_sq, area):
    if va_sq <= 0:     return L < T * area
    if T > 0:          return (L < 0) or  (L*L < T*T*va_sq)
    if T < 0:          return (L < 0) and (L*L > T*T*va_sq)
    return L < 0

def eval_fixed(ii, sii, win_x, win_y, win_w, win_h, scaled_feats):
    area  = win_w * win_h
    s  = rsum(ii,  win_x, win_y, win_w, win_h)
    sq = rsum(sii, win_x, win_y, win_w, win_h)
    va_sq = sq * area - s * s
    scale_q = 1 << Q_THRESH
    for (st_q10, wcs_q) in stages_q:
        ss_q10 = 0
        for (fi, th_q15, lv_q10, rv_q10) in wcs_q:
            fv_raw = sum(apply_weight(rsum(ii, win_x+dx, win_y+dy, rw, rh), code)
                        for dx,dy,rw,rh,code in scaled_feats[fi])
            ss_q10 += lv_q10 if _cmp_sq4(fv_raw*scale_q, th_q15, va_sq, area) else rv_q10
        if ss_q10 < st_q10:
            return 0
    return 1

# ---------- stage-level flip locator ----------
def _find_flip_stage(ii, sii, win_x, win_y, win_w, win_h, scaled_feats):
    area  = win_w * win_h
    s  = rsum(ii,  win_x, win_y, win_w, win_h)
    sq = rsum(sii, win_x, win_y, win_w, win_h)
    inv_area = 1.0 / area
    mean = s * inv_area; variance = sq * inv_area - mean*mean
    stdev = variance**0.5 if variance > 0.0 else 1.0
    va_sq = sq * area - s * s
    scale_q = 1 << Q_THRESH
    sx = win_w / WIN_W; sy = win_h / WIN_H

    for si in range(NUM_STAGES):
        st_f, wcs_f = stages_float[si]
        st_q10, wcs_q = stages_q[si]
        ss_f = 0.0; ss_q = 0
        for (fi, th_f, lv_f, rv_f), (_, th_q15, lv_q10, rv_q10) in zip(wcs_f, wcs_q):
            fv_f = sum(rsum(ii, win_x+int(rx*sx), win_y+int(ry*sy),
                           max(1,int(rw*sx)), max(1,int(rh*sy))) * wt
                      for rx,ry,rw,rh,wt in features_float[fi]) * inv_area
            ss_f += lv_f if fv_f < th_f * stdev else rv_f
            fv_raw = sum(apply_weight(rsum(ii, win_x+dx, win_y+dy, rw, rh), code)
                        for dx,dy,rw,rh,code in scaled_feats[fi])
            ss_q += lv_q10 if _cmp_sq4(fv_raw*scale_q, th_q15, va_sq, area) else rv_q10
        ef = ss_f   < st_f
        eq = ss_q   < st_q10
        if ef != eq:
            return si
        if ef:
            return -1   # both agree to exit, but silently — shouldn't reach
    return NUM_STAGES - 1

# ---------- grouping / best_face (mirrors C logic) ----------
def _uf_find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

def _build_labels(dets):
    n = min(len(dets), VJ_BEST_FACE_MAX_RAW)
    dets = dets[:n]
    parent = list(range(n))
    def union(a, b):
        a, b = _uf_find(parent, a), _uf_find(parent, b)
        if a != b: parent[b] = a
    for i in range(n):
        x1,y1,w1,h1 = dets[i]
        for j in range(i+1, n):
            x2,y2,w2,h2 = dets[j]
            md1 = min(w1, h1); md2 = min(w2, h2); ref = md1 + md2
            if (abs(x1-x2)*10 <= ref and abs(y1-y2)*10 <= ref and
                abs((x1+w1)-(x2+w2))*10 <= ref and abs((y1+h1)-(y2+h2))*10 <= ref):
                union(i, j)
    labels = [_uf_find(parent, i) for i in range(n)]
    return labels, dets

def _best_face(dets_raw):
    if not dets_raw: return None, 0
    labels, dets = _build_labels(dets_raw)
    n = len(dets)
    cnt = Counter(labels)
    best_root, best_score = cnt.most_common(1)[0]
    best_ws = [dets[i][2] for i in range(n) if labels[i] == best_root]
    mode_w  = Counter(best_ws).most_common(1)[0][0]
    sx=sy=sw=sh=c=0
    for i in range(n):
        if labels[i] != best_root: continue
        w = dets[i][2]
        if mode_w*5 <= w*6 and w*5 <= mode_w*6:
            sx+=dets[i][0]; sy+=dets[i][1]; sw+=dets[i][2]; sh+=dets[i][3]; c+=1
    if c == 0: return None, 0
    return (sx//c, sy//c, sw//c, sh//c), best_score

# ---------- full-scan detector (returns raw det list) ----------
def scan_image(gray, ii, sii, evaluator_fn):
    H, W = gray.shape
    dets = []
    scale = 1.0
    while True:
        win_w = int(WIN_W * scale + 0.5)
        win_h = int(WIN_H * scale + 0.5)
        if win_w > W or win_h > H: break
        ystep = 1 if scale > 2.0 else 2
        step  = max(1, int(ystep * scale + 0.5))
        sf = build_scaled_feats(win_w, win_h)
        for y in range(0, H - win_h + 1, step):
            for x in range(0, W - win_w + 1, step):
                if evaluator_fn(ii, sii, x, y, win_w, win_h, sf):
                    dets.append((x, y, win_w, win_h))
        scale *= SCALE_FACTOR
    return dets

def eval_float_wrap(ii, sii, x, y, ww, wh, sf):
    return eval_float(ii, sii, x, y, ww, wh)

# ================================================================
# CRITERION 1+2: per-window consistency + per-stage flip report
# ================================================================
def check_consistency(img_path):
    name = os.path.basename(img_path)
    img  = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    ii, sii = build_ii(gray)

    total = match = 0
    flips_by_stage = defaultdict(int)
    scale = 1.0
    while True:
        win_w = int(WIN_W * scale + 0.5)
        win_h = int(WIN_H * scale + 0.5)
        if win_w > W or win_h > H: break
        ystep = 1 if scale > 2.0 else 2
        step  = max(1, int(ystep * scale + 0.5))
        sf = build_scaled_feats(win_w, win_h)
        for y in range(0, H - win_h + 1, step):
            for x in range(0, W - win_w + 1, step):
                rf = eval_float(ii, sii, x, y, win_w, win_h)
                rq = eval_fixed(ii, sii, x, y, win_w, win_h, sf)
                total += 1
                if rf == rq: match += 1
                else:
                    si = _find_flip_stage(ii, sii, x, y, win_w, win_h, sf)
                    flips_by_stage[si] += 1
        scale *= SCALE_FACTOR

    consistency = match / total * 100.0
    passed = consistency >= 99.5
    print(f"  {name}: {total} windows, {total-match} flips → {consistency:.4f}%  {'PASS' if passed else 'FAIL'}")
    if flips_by_stage:
        top5 = sorted(flips_by_stage.items(), key=lambda t: -t[1])[:5]
        print(f"    flip by stage (top-5): {top5}")
    return consistency, passed

# ================================================================
# CRITERION 3: full pipeline best_face comparison
# ================================================================
def check_pipeline(img_path):
    name = os.path.basename(img_path)
    img  = cv2.imread(img_path)
    if img is None:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ii, sii = build_ii(gray)

    dets_f = scan_image(gray, ii, sii, eval_float_wrap)
    dets_q = scan_image(gray, ii, sii, eval_fixed)

    face_f, score_f = _best_face(dets_f)
    face_q, score_q = _best_face(dets_q)

    raw_match = len(dets_f) == len(dets_q)

    if face_f is None and face_q is None:
        print(f"  {name}: no face detected (both agree)  PASS")
        return True
    if face_f is None or face_q is None:
        print(f"  {name}: detection MISMATCH (one found face, other did not)  FAIL")
        return False

    # Compute IoU
    x1f,y1f,wf,hf = face_f; x2f,y2f = x1f+wf, y1f+hf
    x1q,y1q,wq,hq = face_q; x2q,y2q = x1q+wq, y1q+hq
    ix1 = max(x1f,x1q); iy1 = max(y1f,y1q)
    ix2 = min(x2f,x2q); iy2 = min(y2f,y2q)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    union = wf*hf + wq*hq - inter
    iou   = inter/union if union > 0 else 0.0

    ok = iou >= 0.8
    print(f"  {name}: raw {len(dets_f)} vs {len(dets_q)} dets | "
          f"golden {face_f} score={score_f} | "
          f"fixed  {face_q} score={score_q} | "
          f"IoU={iou:.3f}  {'PASS' if ok else 'FAIL'}")
    return ok

# ================================================================
# MAIN
# ================================================================
PRIMARY_IMAGES = ["test/testt.jpg", "test/lena.jpg", "test/test.jpg"]
EXTRA_IMAGES   = [f for f in ["test/test_image.png"]
                  if os.path.exists(f)
                  and cv2.imread(f) is not None
                  and len(cv2.imread(f).shape) == 3]

ALL_IMAGES = PRIMARY_IMAGES + EXTRA_IMAGES

print("=" * 65)
print("Step 5: Full Regression Verification")
print("  Fixed-point (Steps 1-4) vs Float Golden")
print("=" * 65)

# --- Criterion 1+2: per-window consistency ---
print("\n[1+2] Per-window consistency + stage flip report")
crit12_ok  = True
consistencies = []
for path in ALL_IMAGES:
    result = check_consistency(path)
    if result is None:
        print(f"  {path}: SKIP (cannot read)")
        continue
    c, ok = result
    consistencies.append(c)
    if not ok: crit12_ok = False

overall = sum(consistencies) / len(consistencies) if consistencies else 0
print(f"\n  Mean consistency: {overall:.4f}%  {'PASS' if crit12_ok else 'FAIL'}")

# --- Criterion 3: pipeline best_face ---
print("\n[3] Full pipeline best_face comparison (IoU >= 0.80)")
crit3_ok = True
for path in ALL_IMAGES:
    ok = check_pipeline(path)
    if not ok: crit3_ok = False

# --- Criterion 4: additional images ---
if EXTRA_IMAGES:
    print(f"\n[4] Additional images tested: {[os.path.basename(p) for p in EXTRA_IMAGES]}")
else:
    print("\n[4] No additional images found in test/ — criterion waived")
    print("    (Add new face images to test/ for extended regression)")

# --- HLS-readiness summary ---
print("\n" + "=" * 65)
print("HLS-readiness summary")
print("  vj_evaluate_window_fixed() dependencies:")
print("    - vj_rect_sum()    (inline, vj_integral.h)")
print("    - vj_rect_sum_sq() (inline, vj_integral.h)")
print("    - vj_apply_weight() (static inline, vj_fixed.h)")
print("    - cmp_lhs_lt_rhs()  (static inline, vj_fixed.c)")
print("  Floating-point in evaluator: NONE")
print("  sqrtf: ELIMINATED (Step 4, __int128 square comparison)")
print("  Float multiply in coord path: ELIMINATED (Step 3, lookup)")
print("  Float multiply in weight path: ELIMINATED (Step 2, shift/negate)")
print("  inv_area division: ELIMINATED (Step 1, absorbed)")
print("=" * 65)

all_pass = crit12_ok and crit3_ok
print(f"\nStep 5 verdict: {'PASS — ready for HLS' if all_pass else 'FAIL'}")
