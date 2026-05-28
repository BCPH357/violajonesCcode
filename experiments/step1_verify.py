"""
Step 1 verification: per-window pass/reject consistency between
float golden and fixed-point (inv_area absorbed, Q1.15 threshold, Q6.10 voting).

Consistency threshold: >= 99.5% to pass.
"""
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import sys
from collections import defaultdict

CASCADE_XML = "haarcascade_frontalface_default.xml"
WIN_W = WIN_H = 24
SCALE_FACTOR  = 1.2
Q_THRESH = 15   # threshold Q1.15
Q_LRVAL  = 10   # left/right_val, stage_sum, stage_threshold Q6.10

# ---------- parse cascade ----------
tree = ET.parse(CASCADE_XML)
root = tree.getroot()
cn = root.find(".//cascade")

features_list = []
for feat in cn.find("features").findall("_"):
    rects = []
    for r in feat.find("rects").findall("_"):
        p = r.text.strip().split()
        rects.append((int(p[0]), int(p[1]), int(p[2]), int(p[3]), float(p[4])))
    features_list.append(rects)

stages_float = []   # (stage_thresh_f, [(feat_idx, thresh_f, lv_f, rv_f)])
stages_fixed = []   # (stage_thresh_q10, [(feat_idx, thresh_q15, lv_q10, rv_q10)])

def to_q(v, bits):
    s = v * (1 << bits)
    return int(s + 0.5 if s >= 0 else s - 0.5)

for stage in cn.find("stages").findall("_"):
    st_f = float(stage.find("stageThreshold").text.strip())
    st_q = to_q(st_f, Q_LRVAL)
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
    stages_fixed.append((st_q, wcs_q))

# ---------- integral image ----------
def build_ii(gray):
    g = gray.astype(np.int64)
    ii  = np.zeros((gray.shape[0]+1, gray.shape[1]+1), dtype=np.int64)
    sii = np.zeros((gray.shape[0]+1, gray.shape[1]+1), dtype=np.int64)
    ii[1:, 1:]  = np.cumsum(np.cumsum(g,    axis=0), axis=1)
    sii[1:, 1:] = np.cumsum(np.cumsum(g**2, axis=0), axis=1)
    return ii, sii

def rsum(ii, x, y, w, h):
    return int(ii[y+h, x+w] - ii[y, x+w] - ii[y+h, x] + ii[y, x])

# ---------- float golden evaluator ----------
def eval_float(ii, sii, win_x, win_y, win_w, win_h):
    inv_area = 1.0 / (win_w * win_h)
    s  = rsum(ii,  win_x, win_y, win_w, win_h)
    sq = rsum(sii, win_x, win_y, win_w, win_h)
    mean     = s * inv_area
    variance = sq * inv_area - mean * mean
    stdev    = variance**0.5 if variance > 0.0 else 1.0

    sx = win_w / WIN_W
    sy = win_h / WIN_H

    for (st_f, wcs_f) in stages_float:
        ss = 0.0
        for (fi, th_f, lv_f, rv_f) in wcs_f:
            fv = 0.0
            for (rx, ry, rw, rh, wt) in features_list[fi]:
                ox = win_x + int(rx * sx)
                oy = win_y + int(ry * sy)
                ow = max(1, int(rw * sx))
                oh = max(1, int(rh * sy))
                fv += rsum(ii, ox, oy, ow, oh) * wt
            fv *= inv_area
            ss += lv_f if fv < th_f * stdev else rv_f
        if ss < st_f:
            return 0
    return 1

# ---------- fixed-point evaluator (Step 1) ----------
def eval_fixed(ii, sii, win_x, win_y, win_w, win_h):
    area     = win_w * win_h
    inv_area = 1.0 / area
    s  = rsum(ii,  win_x, win_y, win_w, win_h)
    sq = rsum(sii, win_x, win_y, win_w, win_h)
    mean     = s * inv_area
    variance = sq * inv_area - mean * mean
    stdev    = variance**0.5 if variance > 0.0 else 1.0

    stdev_x_area = stdev * area   # inv_area absorbed

    sx = win_w / WIN_W
    sy = win_h / WIN_H
    scale_q = 1 << Q_THRESH   # 32768

    for (st_q10, wcs_q) in stages_fixed:
        ss_q10 = 0   # int, Q6.10
        for (fi, th_q15, lv_q10, rv_q10) in wcs_q:
            fv_raw = 0
            for (rx, ry, rw, rh, wt) in features_list[fi]:
                ox = win_x + int(rx * sx)
                oy = win_y + int(ry * sy)
                ow = max(1, int(rw * sx))
                oh = max(1, int(rh * sy))
                fv_raw += rsum(ii, ox, oy, ow, oh) * int(wt)  # wt ∈ {-1,2,3}
            # comparison: fv_raw * 32768 < th_q15 * stdev_x_area
            lhs = fv_raw * scale_q
            rhs = th_q15 * stdev_x_area  # float (stdev_x_area still float)
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

    total = match = 0
    flips_by_stage = defaultdict(int)   # stage index → flip count

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
                r_f = eval_float(ii, sii, x, y, win_w, win_h)
                r_q = eval_fixed(ii, sii, x, y, win_w, win_h)
                total += 1
                if r_f == r_q:
                    match += 1
                else:
                    # find which stage caused the flip (diagnostic)
                    _find_flip_stage(ii, sii, x, y, win_w, win_h, flips_by_stage)
        scale *= SCALE_FACTOR

    consistency = match / total * 100.0
    print(f"  {name}: {total} windows, {match} match, {total-match} flip"
          f" → consistency {consistency:.4f}%"
          f"  {'PASS' if consistency >= 99.5 else 'FAIL'}")
    if flips_by_stage:
        top = sorted(flips_by_stage.items(), key=lambda x: -x[1])[:5]
        print(f"    flip by stage (top-5): {top}")
    return consistency

def _find_flip_stage(ii, sii, win_x, win_y, win_w, win_h, acc):
    """Run both versions stage-by-stage to find the first divergence stage."""
    area     = win_w * win_h
    inv_area = 1.0 / area
    s  = rsum(ii,  win_x, win_y, win_w, win_h)
    sq = rsum(sii, win_x, win_y, win_w, win_h)
    mean     = s * inv_area
    variance = sq * inv_area - mean * mean
    stdev    = variance**0.5 if variance > 0.0 else 1.0
    stdev_x_area = stdev * area

    sx = win_w / WIN_W
    sy = win_h / WIN_H
    scale_q = 1 << Q_THRESH

    for stage_idx in range(len(stages_float)):
        st_f, wcs_f = stages_float[stage_idx]
        st_q10, wcs_q = stages_fixed[stage_idx]
        ss_f  = 0.0
        ss_q10 = 0

        for i, ((fi, th_f, lv_f, rv_f), (_, th_q15, lv_q10, rv_q10)) in \
                enumerate(zip(wcs_f, wcs_q)):
            fv_raw = 0
            for (rx, ry, rw, rh, wt) in features_list[fi]:
                ox = win_x + int(rx * sx); oy = win_y + int(ry * sy)
                ow = max(1, int(rw * sx)); oh = max(1, int(rh * sy))
                fv_raw += rsum(ii, ox, oy, ow, oh) * int(wt)
            fv_norm = fv_raw * inv_area
            ss_f   += lv_f   if fv_norm < th_f * stdev   else rv_f
            lhs = fv_raw * scale_q
            rhs = th_q15 * stdev_x_area
            ss_q10 += lv_q10 if lhs < rhs else rv_q10

        # check if this stage gives different exit decision
        exit_f  = ss_f   < st_f
        exit_q  = ss_q10 < st_q10
        if exit_f != exit_q:
            acc[stage_idx] += 1
            return
        if exit_f:   # both agree to exit here
            return

print("=== Step 1 Verification: float golden vs fixed-point (Q1.15/Q6.10) ===")
print(f"    threshold: Q1.15 ({1<<Q_THRESH} scale)")
print(f"    left/right_val, stage_sum, stage_threshold: Q6.10 ({1<<Q_LRVAL} scale)")
print(f"    inv_area: absorbed (stdev×area pre-computed)")
print(f"    sqrtf: kept (Step 4 will eliminate)")
print()

results = {}
for name in ["testt.jpg", "lena.jpg", "test.jpg"]:
    results[name] = run_image(name)

print()
overall = sum(results.values()) / len(results)
print(f"Mean consistency across images: {overall:.4f}%")
verdict = "PASS" if all(v >= 99.5 for v in results.values()) else "FAIL"
print(f"Step 1 verdict: {verdict}")
