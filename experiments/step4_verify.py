"""
Step 4 verification: eliminate sqrtf via four-branch sign-safe square comparison.

L  = feat_val_raw * 2^15          (int, can be any sign)
R  = threshold_q15 * sqrt(va_sq)  (va_sq = sqsum*area - sum^2 = variance*area^2)
va_sq computed in int64; L^2 / R^2 need arbitrary precision (int128 in C).

Four branches (stdev != 0 path):
  T > 0  (R >= 0):
    L <  0  -> True  (negative < non-negative)
    L >= 0  -> L^2 < T^2 * va_sq
  T < 0  (R <= 0):
    L >= 0  -> False (non-negative >= non-positive)
    L <  0  -> L^2 > T^2 * va_sq  (both negative: more negative = smaller)
  T == 0  (R == 0): L < 0

Special case (va_sq <= 0 <-> variance <= 0):
  float golden defaults stdev=1.0, stdev_x_area=area
  -> comparison becomes L < T * area  (pure int64)

Checks:
  A) Step4 vs Step3  (sqrtf reference)  -> expect >= 99.5%
  B) Step4 vs golden (float golden)     -> expect >= 99.5%
"""
import xml.etree.ElementTree as ET
import numpy as np
import cv2

CASCADE_XML  = "haarcascade_frontalface_default.xml"
WIN_W = WIN_H = 24
SCALE_FACTOR  = 1.2
Q_THRESH = 15
Q_LRVAL  = 10

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
        rf.append((x, y, w, h, wt))
        rc.append((x, y, w, h, weight_to_code(wt)))
    features_float.append(rf)
    features_code.append(rc)

def to_q(v, bits):
    s = v * (1 << bits)
    return int(s + 0.5 if s >= 0 else s - 0.5)

stages_float = []
stages_q     = []
for stage in cn.find("stages").findall("_"):
    st_f  = float(stage.find("stageThreshold").text.strip())
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

# ---------- Step 3: precomputed coordinate table ----------
def build_scaled_feats(win_w, win_h):
    sx = win_w / WIN_W; sy = win_h / WIN_H
    table = []
    for rects in features_code:
        scaled = []
        for (x, y, w, h, code) in rects:
            dx = int(x * sx); dy = int(y * sy)
            rw = max(1, int(w * sx)); rh = max(1, int(h * sy))
            scaled.append((dx, dy, rw, rh, code))
        table.append(scaled)
    return table

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

# ---------- Step 3 (sqrtf reference) ----------
def eval_step3(ii, sii, win_x, win_y, win_w, win_h, scaled_feats):
    area     = win_w * win_h
    inv_area = 1.0 / area
    s  = rsum(ii,  win_x, win_y, win_w, win_h)
    sq = rsum(sii, win_x, win_y, win_w, win_h)
    mean         = s * inv_area
    variance     = sq * inv_area - mean * mean
    stdev        = variance**0.5 if variance > 0.0 else 1.0
    stdev_x_area = stdev * area
    scale_q = 1 << Q_THRESH
    for (st_q10, wcs_q) in stages_q:
        ss_q10 = 0
        for (fi, th_q15, lv_q10, rv_q10) in wcs_q:
            fv_raw = sum(apply_weight(rsum(ii, win_x+dx, win_y+dy, rw, rh), code)
                        for dx,dy,rw,rh,code in scaled_feats[fi])
            lhs = fv_raw * scale_q
            rhs = th_q15 * stdev_x_area
            ss_q10 += lv_q10 if lhs < rhs else rv_q10
        if ss_q10 < st_q10:
            return 0
    return 1

# ---------- Step 4: integer four-branch, no sqrtf ----------
def _cmp_sq4(L, T, va_sq, area):
    """
    Integer comparison L < T * sqrt(va_sq).
    When va_sq <= 0 (variance <= 0): use default stdev=1, compare L < T*area.
    All arithmetic is Python arbitrary-precision (equivalent to __int128 in C).
    """
    if va_sq <= 0:
        return L < T * area
    if T > 0:
        return (L < 0) or (L * L < T * T * va_sq)
    if T < 0:
        return (L < 0) and (L * L > T * T * va_sq)
    return L < 0   # T == 0: R = 0

def eval_step4(ii, sii, win_x, win_y, win_w, win_h, scaled_feats):
    area = win_w * win_h
    s  = rsum(ii,  win_x, win_y, win_w, win_h)
    sq = rsum(sii, win_x, win_y, win_w, win_h)
    # variance * area^2, exact integer (fits int64 in C)
    va_sq = sq * area - s * s

    scale_q = 1 << Q_THRESH
    for (st_q10, wcs_q) in stages_q:
        ss_q10 = 0
        for (fi, th_q15, lv_q10, rv_q10) in wcs_q:
            fv_raw = sum(apply_weight(rsum(ii, win_x+dx, win_y+dy, rw, rh), code)
                        for dx,dy,rw,rh,code in scaled_feats[fi])
            L = fv_raw * scale_q   # Python big int (int64 in C)
            less = _cmp_sq4(L, th_q15, va_sq, area)
            ss_q10 += lv_q10 if less else rv_q10
        if ss_q10 < st_q10:
            return 0
    return 1

# ---------- scan and compare ----------
def run_image(name):
    img  = cv2.imread(f"test/{name}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    ii, sii = build_ii(gray)

    total = s4_vs_s3 = s4_vs_f = 0
    scale = 1.0
    while True:
        win_w = int(WIN_W * scale + 0.5)
        win_h = int(WIN_H * scale + 0.5)
        if win_w > W or win_h > H:
            break
        ystep = 1 if scale > 2.0 else 2
        step  = max(1, int(ystep * scale + 0.5))
        sf = build_scaled_feats(win_w, win_h)

        for y in range(0, H - win_h + 1, step):
            for x in range(0, W - win_w + 1, step):
                r3 = eval_step3(ii, sii, x, y, win_w, win_h, sf)
                r4 = eval_step4(ii, sii, x, y, win_w, win_h, sf)
                rf = eval_float (ii, sii, x, y, win_w, win_h)
                total += 1
                if r4 == r3: s4_vs_s3 += 1
                if r4 == rf: s4_vs_f  += 1
        scale *= SCALE_FACTOR

    c43 = s4_vs_s3 / total * 100.0
    c4f = s4_vs_f  / total * 100.0
    p43 = 'PASS' if c43 >= 99.5 else 'FAIL'
    p4f = 'PASS' if c4f >= 99.5 else 'FAIL'
    print(f"  {name}: {total} windows")
    print(f"    Step4 vs Step3:   {c43:.4f}%  {p43}")
    print(f"    Step4 vs golden:  {c4f:.4f}%  {p4f}")
    if c43 < 100.0:
        flips = total - s4_vs_s3
        print(f"    (sqrt elimination flips vs Step3: {flips} windows)")
    return c43, c4f

print("=== Step 4 Verification: eliminate sqrtf (four-branch square comparison) ===")
print("    L = feat_val_raw * 2^15,  va_sq = sqsum*area - sum^2")
print("    C impl: va_sq in int64, L^2/R^2 in __int128")
print()

results = {}
for name in ["testt.jpg", "lena.jpg", "test.jpg"]:
    results[name] = run_image(name)

print()
avg43 = sum(v[0] for v in results.values()) / len(results)
avg4f = sum(v[1] for v in results.values()) / len(results)
print(f"Mean Step4 vs Step3:  {avg43:.4f}%")
print(f"Mean Step4 vs golden: {avg4f:.4f}%")
ok = all(v[1] >= 99.5 for v in results.values())
print(f"Step 4 verdict: {'PASS' if ok else 'FAIL'}")
