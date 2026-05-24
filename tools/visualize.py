#!/usr/bin/env python3
"""
Visualize detection results: OpenCV vs our C implementation, side by side.
Usage: python tools/visualize.py <image_path> [min_neighbors] [scale_factor]
"""

import sys, os, subprocess
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" or "Agg" if TkAgg fails
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)
CASCADE_XML = os.path.join(ROOT_DIR, "haarcascade_frontalface_default.xml")
TEST_EXE    = os.path.join(ROOT_DIR, "test", "test_vj.exe")
TMP_RAW     = os.path.join(ROOT_DIR, "test", "_tmp_gray.raw")

# ---------- helpers ----------

def to_gray(img_path):
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot open {img_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def run_opencv(gray, scale_factor, min_neighbors):
    cc = cv2.CascadeClassifier(CASCADE_XML)
    faces = cc.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(24, 24),
    )
    return [] if len(faces) == 0 else [(int(x), int(y), int(w), int(h)) for x,y,w,h in faces]


def run_c(gray, scale_factor, min_neighbors, best_only=False):
    h, w = gray.shape
    with open(TMP_RAW, "wb") as f:
        f.write(gray.tobytes())
    mn_arg = "best" if best_only else str(min_neighbors)
    cmd = [TEST_EXE, TMP_RAW, str(w), str(h),
           f"{scale_factor:.2f}", mn_arg]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("C detector stderr:", result.stderr)
        return []
    dets = []
    for line in result.stdout.splitlines():
        if line.strip().startswith("["):
            parts = line.split()
            d = {}
            for p in parts:
                if "=" in p:
                    k, v = p.split("=")
                    d[k] = int(v)
            if "x" in d:
                dets.append((d["x"], d["y"], d["w"], d["h"]))
    return dets


def draw_boxes(ax, img_rgb, dets, title, color):
    ax.imshow(img_rgb)
    ax.set_title(f"{title}\n({len(dets)} detection{'s' if len(dets)!=1 else ''})",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
    for (x, y, w, h) in dets:
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x, y - 4, f"{w}×{h}", color=color, fontsize=7,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))


# ---------- main ----------

def main():
    img_path      = sys.argv[1] if len(sys.argv) > 1 else None
    min_neighbors = int(sys.argv[2])   if len(sys.argv) > 2 else 3
    scale_factor  = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2

    if img_path is None:
        print(f"Usage: python {__file__} <image> [min_neighbors] [scale_factor]")
        sys.exit(1)

    if not os.path.exists(TEST_EXE):
        print(f"ERROR: {TEST_EXE} not found. Run build.bat first.")
        sys.exit(1)

    print(f"Image:         {img_path}")
    print(f"Scale factor:  {scale_factor}")
    print(f"Min neighbors: {min_neighbors}")

    gray = to_gray(img_path)
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    h, w = gray.shape
    print(f"Size:          {w}x{h}")

    print("Running OpenCV...")
    opencv_dets = run_opencv(gray, scale_factor, min_neighbors)
    print(f"  Found {len(opencv_dets)} face(s)")

    print("Running C detector (best face only)...")
    c_dets = run_c(gray, scale_factor, min_neighbors, best_only=True)
    print(f"  Found {len(c_dets)} face(s)")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"{os.path.basename(img_path)}  |  scaleFactor={scale_factor}  minNeighbors={min_neighbors}",
        fontsize=13, fontweight="bold"
    )

    draw_boxes(axes[0], img_rgb, opencv_dets, "OpenCV CascadeClassifier", "#00FF00")
    draw_boxes(axes[1], img_rgb, c_dets,      "Pure-C Viola-Jones — Best Face (MCU)", "#FF4444")

    plt.tight_layout()

    # also save PNG
    out_path = os.path.splitext(img_path)[0] + "_detection.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
