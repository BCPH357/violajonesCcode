"""Shared utilities for Viola-Jones experiments."""
import os
import sys
import subprocess
import time
import urllib.request

import cv2
import numpy as np

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASCADE_XML = os.path.join(ROOT_DIR, "haarcascade_frontalface_default.xml")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "experiments", "outputs")

# C binaries live in MSYS2 home to bypass Application Control policies
VJ_EXE    = r"C:\msys64\home\miat\test_vj.exe"
BENCH_EXE = r"C:\msys64\home\miat\bench_vj.exe"

TMP_RAW = os.path.join(ROOT_DIR, "test", "_exp_tmp.raw")

YUNET_MODEL = os.path.join(ROOT_DIR, "experiments", "face_detection_yunet_2023mar.onnx")
YUNET_URL   = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def ensure_outputs_dir():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ---------- IoU helpers ----------

def compute_iou(a, b):
    """IoU between two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx);  iy1 = max(ay, by)
    ix2 = min(ax+aw, bx+bw);  iy2 = min(ay+ah, by+bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


def match_greedy(ref_dets, pred_dets, iou_thresh=0.5):
    """
    Greedy one-to-one matching: pair each pred to the best unmatched ref.
    Returns (tp, fp, fn, matched_ious).
    """
    ref_used = [False] * len(ref_dets)
    matched_ious = []
    tp = fp = 0

    for pred in pred_dets:
        best_iou, best_j = 0.0, -1
        for j, ref in enumerate(ref_dets):
            if ref_used[j]:
                continue
            iou = compute_iou(pred, ref)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thresh:
            tp += 1
            ref_used[best_j] = True
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = sum(1 for u in ref_used if not u)
    return tp, fp, fn, matched_ious


# ---------- detectors ----------

def run_opencv(gray_img, scale_factor=1.2, min_neighbors=3):
    """Run OpenCV CascadeClassifier, return list of (x,y,w,h)."""
    cc = cv2.CascadeClassifier(CASCADE_XML)
    faces = cc.detectMultiScale(
        gray_img,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(24, 24),
    )
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def _parse_c_output(stdout):
    dets = []
    for line in stdout.splitlines():
        if line.strip().startswith("["):
            d = {}
            for p in line.split():
                if "=" in p:
                    k, v = p.split("=", 1)
                    d[k] = int(v)
            if "x" in d:
                dets.append((d["x"], d["y"], d["w"], d["h"]))
    return dets


def run_c_detect(gray_img, scale_factor=1.2, min_neighbors=3):
    """Run C Viola-Jones detector. Returns list of (x,y,w,h) or None if unavailable."""
    if not os.path.exists(VJ_EXE):
        return None
    h, w = gray_img.shape
    with open(TMP_RAW, "wb") as f:
        f.write(gray_img.tobytes())
    cmd = [VJ_EXE, TMP_RAW, str(w), str(h),
           f"{scale_factor:.2f}", str(min_neighbors)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return _parse_c_output(r.stdout)


def run_c_best_face(gray_img, scale_factor=1.2):
    """Run C VJ in best-face mode (vj_detect_best_face). Returns [(x,y,w,h)] or [] or None."""
    if not os.path.exists(VJ_EXE):
        return None
    h, w = gray_img.shape
    with open(TMP_RAW, "wb") as f:
        f.write(gray_img.tobytes())
    cmd = [VJ_EXE, TMP_RAW, str(w), str(h), f"{scale_factor:.2f}", "best"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return _parse_c_output(r.stdout)


def run_c_bench(gray_img, scale_factor=1.2, min_neighbors=3, n_runs=50, best_mode=False):
    """Run bench_vj.exe and return timing+stats dict, or None if unavailable."""
    if not os.path.exists(BENCH_EXE):
        return None
    h, w = gray_img.shape
    with open(TMP_RAW, "wb") as f:
        f.write(gray_img.tobytes())
    mn_arg = "best" if best_mode else str(min_neighbors)
    cmd = [BENCH_EXE, TMP_RAW, str(w), str(h),
           f"{scale_factor:.2f}", mn_arg, str(n_runs)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None
    data = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                data[k.strip()] = float(v.strip())
            except ValueError:
                data[k.strip()] = v.strip()
    return data if data else None


def ensure_yunet_model():
    """Download YuNet ONNX model if not present. Return True on success."""
    if os.path.exists(YUNET_MODEL):
        return True
    print("  Downloading YuNet model (~350 KB)...")
    try:
        req = urllib.request.Request(YUNET_URL,
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(YUNET_MODEL, "wb") as f:
            f.write(data)
        print(f"  Saved: {YUNET_MODEL}")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def run_yunet(bgr_img, conf_threshold=0.5):
    """Run YuNet face detector. Returns list of (x,y,w,h) or None if unavailable."""
    if not ensure_yunet_model():
        return None
    h, w = bgr_img.shape[:2]
    det = cv2.FaceDetectorYN.create(
        YUNET_MODEL, "", (w, h),
        score_threshold=conf_threshold,
        nms_threshold=0.3,
        top_k=5000,
    )
    _, faces = det.detect(bgr_img)
    if faces is None:
        return []
    return [(int(face[0]), int(face[1]), int(face[2]), int(face[3])) for face in faces]


# ---------- timing ----------

def time_opencv(gray_img, n_runs=50, scale_factor=1.2, min_neighbors=3):
    """Time OpenCV CascadeClassifier (wall-clock). Returns dict with mean/std/min/max ms."""
    cc = cv2.CascadeClassifier(CASCADE_XML)
    for _ in range(3):  # warm up
        cc.detectMultiScale(gray_img, scaleFactor=scale_factor,
                            minNeighbors=min_neighbors, minSize=(24, 24))
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        cc.detectMultiScale(gray_img, scaleFactor=scale_factor,
                            minNeighbors=min_neighbors, minSize=(24, 24))
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return {"mean_ms": a.mean(), "std_ms": a.std(),
            "min_ms": a.min(), "max_ms": a.max(), "n_runs": n_runs}


def time_yunet(bgr_img, n_runs=50, conf_threshold=0.5):
    """Time YuNet (wall-clock). Returns timing dict or None."""
    if not ensure_yunet_model():
        return None
    h, w = bgr_img.shape[:2]
    det = cv2.FaceDetectorYN.create(
        YUNET_MODEL, "", (w, h),
        score_threshold=conf_threshold,
        nms_threshold=0.3,
        top_k=5000,
    )
    for _ in range(3):
        det.detect(bgr_img)
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        det.detect(bgr_img)
        ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return {"mean_ms": a.mean(), "std_ms": a.std(),
            "min_ms": a.min(), "max_ms": a.max(), "n_runs": n_runs}


# ---------- image helpers ----------

def get_test_images():
    """Return sorted list of test image paths (exclude generated output images)."""
    test_dir = os.path.join(ROOT_DIR, "test")
    skip = {"test_image.png", "test_detection.png", "testt_detection.png"}
    exts = (".jpg", ".jpeg", ".png")
    imgs = []
    for f in sorted(os.listdir(test_dir)):
        if f.lower().endswith(exts) and not f.startswith("_") and f not in skip:
            imgs.append(os.path.join(test_dir, f))
    return imgs


def load_image(path):
    """Return (bgr, gray) tuple, or (None, None) on failure."""
    bgr = cv2.imread(path)
    if bgr is None:
        return None, None
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def compute_window_count(img_w, img_h, win_w=24, win_h=24, scale_factor=1.2):
    """Compute total sliding-window candidates (before cascade filtering)."""
    count = 0
    scale = 1.0
    while True:
        ww = int(win_w * scale)
        wh = int(win_h * scale)
        if ww > img_w or wh > img_h:
            break
        step = max(1, int(scale * 2.0 + 0.5))
        nx = (img_w - ww) // step + 1
        ny = (img_h - wh) // step + 1
        if nx > 0 and ny > 0:
            count += nx * ny
        scale *= scale_factor
    return count
