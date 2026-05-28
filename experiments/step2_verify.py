"""
Step 2 verification: weight float multiply → shift/negate (VJ_WCODE_*).

Checks:
  A) Step 2 vs Step 1 (should be exactly 100% — only weight path changed)
  B) Step 2 vs float golden (same as Step 1 result, expect 100%)

Consistency threshold: >= 99.5% to pass.
"""
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import sys

CASCADE_XML = "haarcascade_frontalface_default.xml"
WIN_W = WIN_H = 24
SCALE_FACTOR  = 1.2
Q_THRESH = 15
Q_LRVAL  = 10

# ---------- parse cascade ----------
tree = ET.parse(CASCADE_XML)
root = tree.getroot()
cn   = root.find(".//cascade")

WCODE_NEG1 = 0
WCODE_POS2 = 1
WCODE_POS3 = 2

def weight_to_code(w):
    if w < 0:   return WCODE_NEG1
    if w < 2.5: return WCODE_POS2
    return WCODE_POS3

def apply_weight(rs, code):
    if code == WCODE_NEG1: return -rs
    if code == WCODE_POS2: return rs << 1
    return (rs << 1) + rs   # WCODE_POS3

# float rects: (x,y,w,h, weight_float)
# fixed rects: (x,y,w,h, weight_code)
features_float = []
features_fixed = []
for feat in cn.find("features").findall("_"):
    rf, rc2 = [], []
    for r in feat.find("rects").findall("_"):
        p = r.text.strip().split()
        x,y,w,h,wt = int(p[0]),int(p[1]),int(p[2]),int(p[3]),float(p[4])
        rf.append((x, y, w, h, wt))
        rc2.append((x, y, w, h, weight_to_code(wt)))
    features_float.append(rf)
    features_fixed.append(rc2)

def to_q(v, bits):
    s = v * (1 << bits)
    return int(s + 0.5 if s >= 0 else s - 0.5)

stages_float = []
stages_step1  = []   # Step 1: uses features_float, weight as int
stages_step2  = []   # Step 2: uses features_fixed (weight_code)

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
    stages_step1.append((st_q, wcs_q))
    stages_step2.append((st_q, wcs_q))   # same Q values; feature path differs in eval

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

# ---------- Step 1 (reference) ----------
def eval_step1(ii, sii, win_x, win_y, win_w, win_h):
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
    for (st_q10, wcs_q) in stages_step1:
        ss_q10 = 0
        for (fi, th_q15, lv_q10, rv_q10) in wcs_q:
            fv_raw = 0
            for (rx,ry,rw,rh,wt) in features_float[fi]:  # float weight as int
                ox=win_x+int(rx*sx); oy=win_y+int(ry*sy)
                ow=max(1,int(rw*sx)); oh=max(1,int(rh*sy))
                fv_raw += rsum(ii, ox, oy, ow, oh) * int(wt)
            lhs = fv_raw * scale_q
            rhs = th_q15 * stdev_x_area
            ss_q10 += lv_q10 if lhs < rhs else rv_q10
        if ss_q10 < st_q10:
            return 0
    return 1

# ---------- Step 2 (weight codes) ----------
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
    for (st_q10, wcs_q) in stages_step2:
        ss_q10 = 0
        for (fi, th_q15, lv_q10, rv_q10) in wcs_q:
            fv_raw = 0
            for (rx,ry,rw,rh,code) in features_fixed[fi]:  # weight_code
                ox=win_x+int(rx*sx); oy=win_y+int(ry*sy)
                ow=max(1,int(rw*sx)); oh=max(1,int(rh*sy))
                rs = rsum(ii, ox, oy, ow, oh)
                fv_raw += apply_weight(rs, code)
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

    total = s1_match = s2_match = s2_vs_f_match = 0
    scale = 1.0
    while True:
        win_w = int(WIN_W * scale + 0.5)
        win_h = int(WIN_H * scale + 0.5)
        if win_w > W or win_h > H:
            break
        ystep = 1 if scale > 2.0 else 2
        step  = max(1, int(ystep * scale + 0.5))

        for y in range(0, H - win_h + 1, step):
            for x in range(0, W - win_w + 1, step):
                r1 = eval_step1(ii, sii, x, y, win_w, win_h)
                r2 = eval_step2(ii, sii, x, y, win_w, win_h)
                rf = eval_float (ii, sii, x, y, win_w, win_h)
                total += 1
                if r1 == r2:           s1_match += 1
                if r2 == r1:           s2_match += 1    # same as s1_match
                if r2 == rf:           s2_vs_f_match += 1
        scale *= SCALE_FACTOR

    c12 = s1_match / total * 100.0
    c2f = s2_vs_f_match / total * 100.0
    p12 = 'PASS' if c12 >= 99.5 else 'FAIL'
    p2f = 'PASS' if c2f >= 99.5 else 'FAIL'
    print(f"  {name}: {total} windows")
    print(f"    Step2 vs Step1:   {c12:.4f}%  {p12}")
    print(f"    Step2 vs golden:  {c2f:.4f}%  {p2f}")
    return c12, c2f

print("=== Step 2 Verification: weight float→int vs shift/negate ===")
print(f"    weight codes: -1→NEG, +2→POS2, +3→POS3  (shift/negate only)")
print()

results = {}
for name in ["testt.jpg", "lena.jpg", "test.jpg"]:
    results[name] = run_image(name)

print()
avg12 = sum(v[0] for v in results.values()) / len(results)
avg2f = sum(v[1] for v in results.values()) / len(results)
print(f"Mean Step2 vs Step1:  {avg12:.4f}%")
print(f"Mean Step2 vs golden: {avg2f:.4f}%")
ok12 = all(v[0] >= 99.5 for v in results.values())
ok2f = all(v[1] >= 99.5 for v in results.values())
print(f"Step 2 verdict: {'PASS' if ok12 and ok2f else 'FAIL'}")
