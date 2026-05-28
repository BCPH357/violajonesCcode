"""
Step 3 verification: coord scaling float multiply → precomputed lookup table.

Route A: precompute (dx, dy, rw, rh) per feature per scale level offline.
         Runtime: table lookup only, zero float multiply for coordinates.

Checks:
  A) Step3 vs Step2  → expect ≈ 100% (同樣取整規則，僅「計算時機」不同)
  B) Step3 vs golden → expect ≈ 100% (取整規則一致，無近似誤差)

Consistency threshold: >= 99.5% to pass.
"""
import xml.etree.ElementTree as ET
import numpy as np
import cv2

CASCADE_XML = "haarcascade_frontalface_default.xml"
WIN_W = WIN_H = 24
SCALE_FACTOR  = 1.2
Q_THRESH = 15
Q_LRVAL  = 10

# ---------- parse cascade ----------
tree = ET.parse(CASCADE_XML)
cn   = tree.getroot().find(".//cascade")

WCODE_NEG1, WCODE_POS2, WCODE_POS3 = 0, 1, 2

def weight_to_code(w):
    if w < 0:   return WCODE_NEG1
    if w < 2.5: return WCODE_POS2
    return WCODE_POS3

def apply_weight(rs, code):
    if code == WCODE_NEG1: return -rs
    if code == WCODE_POS2: return rs << 1
    return (rs << 1) + rs

# features_float: (x,y,w,h, weight_float) — for float golden & Step2
# features_code:  (x,y,w,h, weight_code)  — for Step2 (weight codes only)
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
stages_q     = []   # shared by Step2 and Step3 (same Q values)

for stage in cn.find("stages").findall("_"):
    st_f  = float(stage.find("stageThreshold").text.strip())
    st_q  = to_q(st_f, Q_LRVAL)
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
    stages_q.append((st_q, wcs_q))

# ---------- integral image ----------
def build_ii(gray):
    g  = gray.astype(np.int64)
    ii  = np.zeros((gray.shape[0]+1, gray.shape[1]+1), dtype=np.int64)
    sii = np.zeros((gray.shape[0]+1, gray.shape[1]+1), dtype=np.int64)
    ii[1:, 1:]  = np.cumsum(np.cumsum(g,    axis=0), axis=1)
    sii[1:, 1:] = np.cumsum(np.cumsum(g**2, axis=0), axis=1)
    return ii, sii

def rsum(ii, x, y, w, h):
    return int(ii[y+h, x+w] - ii[y, x+w] - ii[y+h, x] + ii[y, x])

# ---------- Step 3: precomputed coordinate table ----------
def build_scaled_feats(win_w, win_h):
    """Offline precompute (dx,dy,rw,rh) for every feature at this scale.
    Uses identical truncation as float golden: int(x * scale_x) = floor."""
    sx = win_w / WIN_W
    sy = win_h / WIN_H
    table = []
    for rects in features_code:
        scaled = []
        for (x, y, w, h, code) in rects:
            dx = int(x * sx)             # floor, same as C (int)(x*sx)
            dy = int(y * sy)
            rw = max(1, int(w * sx))
            rh = max(1, int(h * sy))
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
            fv = 0.0
            for (rx,ry,rw,rh,wt) in features_float[fi]:
                ox=win_x+int(rx*sx); oy=win_y+int(ry*sy)
                ow=max(1,int(rw*sx)); oh=max(1,int(rh*sy))
                fv += rsum(ii, ox, oy, ow, oh) * wt
            fv *= inv_area
            ss += lv_f if fv < th_f * stdev else rv_f
        if ss < st_f:
            return 0
    return 1

# ---------- Step 2 (reference, runtime float multiply for coords) ----------
def eval_step2(ii, sii, win_x, win_y, win_w, win_h):
    area     = win_w * win_h
    inv_area = 1.0 / area
    s  = rsum(ii,  win_x, win_y, win_w, win_h)
    sq = rsum(sii, win_x, win_y, win_w, win_h)
    mean     = s * inv_area
    variance = sq * inv_area - mean * mean
    stdev    = variance**0.5 if variance > 0.0 else 1.0
    stdev_x_area = stdev * area
    sx = win_w / WIN_W; sy = win_h / WIN_H
    scale_q = 1 << Q_THRESH
    for (st_q10, wcs_q) in stages_q:
        ss_q10 = 0
        for (fi, th_q15, lv_q10, rv_q10) in wcs_q:
            fv_raw = 0
            for (rx,ry,rw,rh,code) in features_code[fi]:
                ox=win_x+int(rx*sx); oy=win_y+int(ry*sy)
                ow=max(1,int(rw*sx)); oh=max(1,int(rh*sy))
                fv_raw += apply_weight(rsum(ii, ox, oy, ow, oh), code)
            lhs = fv_raw * scale_q
            rhs = th_q15 * stdev_x_area
            ss_q10 += lv_q10 if lhs < rhs else rv_q10
        if ss_q10 < st_q10:
            return 0
    return 1

# ---------- Step 3 (precomputed lookup, no runtime float coord multiply) ----------
def eval_step3(ii, sii, win_x, win_y, win_w, win_h, scaled_feats):
    area     = win_w * win_h
    inv_area = 1.0 / area
    s  = rsum(ii,  win_x, win_y, win_w, win_h)
    sq = rsum(sii, win_x, win_y, win_w, win_h)
    mean     = s * inv_area
    variance = sq * inv_area - mean * mean
    stdev    = variance**0.5 if variance > 0.0 else 1.0
    stdev_x_area = stdev * area
    scale_q = 1 << Q_THRESH
    for (st_q10, wcs_q) in stages_q:
        ss_q10 = 0
        for (fi, th_q15, lv_q10, rv_q10) in wcs_q:
            fv_raw = 0
            for (dx, dy, rw, rh, code) in scaled_feats[fi]:  # table lookup
                rx = win_x + dx
                ry = win_y + dy
                fv_raw += apply_weight(rsum(ii, rx, ry, rw, rh), code)
            lhs = fv_raw * scale_q
            rhs = th_q15 * stdev_x_area
            ss_q10 += lv_q10 if lhs < rhs else rv_q10
        if ss_q10 < st_q10:
            return 0
    return 1

# ---------- scan and compare ----------
def run_image(name):
    img  = cv2.imread(f"test/{name}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    ii, sii = build_ii(gray)

    total = s3_vs_s2 = s3_vs_f = 0
    scale = 1.0
    while True:
        win_w = int(WIN_W * scale + 0.5)
        win_h = int(WIN_H * scale + 0.5)
        if win_w > W or win_h > H:
            break
        ystep = 1 if scale > 2.0 else 2
        step  = max(1, int(ystep * scale + 0.5))

        # Build lookup table once per scale level (O(num_features) offline work)
        scaled_feats = build_scaled_feats(win_w, win_h)

        for y in range(0, H - win_h + 1, step):
            for x in range(0, W - win_w + 1, step):
                r2 = eval_step2(ii, sii, x, y, win_w, win_h)
                r3 = eval_step3(ii, sii, x, y, win_w, win_h, scaled_feats)
                rf = eval_float (ii, sii, x, y, win_w, win_h)
                total += 1
                if r3 == r2: s3_vs_s2 += 1
                if r3 == rf: s3_vs_f  += 1
        scale *= SCALE_FACTOR

    c32 = s3_vs_s2 / total * 100.0
    c3f = s3_vs_f  / total * 100.0
    p32 = 'PASS' if c32 >= 99.5 else 'FAIL'
    p3f = 'PASS' if c3f >= 99.5 else 'FAIL'
    print(f"  {name}: {total} windows")
    print(f"    Step3 vs Step2:   {c32:.4f}%  {p32}")
    print(f"    Step3 vs golden:  {c3f:.4f}%  {p3f}")
    return c32, c3f

print("=== Step 3 Verification: coord scaling float → precomputed lookup ===")
print(f"    Route A: offline build_scaled_feats() per scale level")
print(f"    取整規則: int(x*sx) = floor  (一致於 float golden)")
print()

results = {}
for name in ["testt.jpg", "lena.jpg", "test.jpg"]:
    results[name] = run_image(name)

print()
avg32 = sum(v[0] for v in results.values()) / len(results)
avg3f = sum(v[1] for v in results.values()) / len(results)
print(f"Mean Step3 vs Step2:  {avg32:.4f}%")
print(f"Mean Step3 vs golden: {avg3f:.4f}%")
ok32 = all(v[0] >= 99.5 for v in results.values())
ok3f = all(v[1] >= 99.5 for v in results.values())
print(f"Step 3 verdict: {'PASS' if ok32 and ok3f else 'FAIL'}")
