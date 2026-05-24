"""
Experiment 2: Accuracy vs. modern detector — trade-off framing.

Detection mode:
  C VJ  → vj_detect_best_face() — the correct rPPG scenario (single face tracking).
           This bypasses the minNeighbors grouping threshold that caused false negatives
           in Exp1 and instead selects the group with the highest vote count.
  OpenCV → detectMultiScale(minNeighbors=3)  (OpenCV default)
  YuNet  → score_threshold=0.5

Pseudo-GT: same image is "face present" if at least one detector found a face.
           Recall = 1 if the detector found at least one box, 0 otherwise.
           This avoids needing external annotations for the available 3-image set.

Latency: 20-run mean.  C uses bench_vj.exe in best-face mode; others use
          Python perf_counter.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import (
    ROOT_DIR, OUTPUTS_DIR, BENCH_EXE, YUNET_MODEL,
    ensure_outputs_dir, ensure_yunet_model,
    run_opencv, run_c_best_face, run_yunet,
    run_c_bench,
    get_test_images, load_image, CASCADE_XML,
)

SCALE_FACTOR  = 1.2
MIN_NEIGHBORS = 3
IOU_THRESH    = 0.5
N_LAT_RUNS    = 20


def _quick_py_latency(fn, n=20):
    for _ in range(3):
        fn()
    calls = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        calls.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(calls))


def run_experiment():
    print(f"\n=== Experiment 2: Accuracy vs. Cost Trade-off ===")
    print(f"C mode: vj_detect_best_face()  |  OpenCV/YuNet: standard mode")

    images = get_test_images()
    all_rows = []

    for img_path in images:
        bgr, gray = load_image(img_path)
        if gray is None:
            continue
        name = os.path.basename(img_path)
        h, w = gray.shape
        print(f"\n  {name} ({w}x{h}):")

        opencv_dets = run_opencv(gray, SCALE_FACTOR, MIN_NEIGHBORS)
        c_dets      = run_c_best_face(gray, SCALE_FACTOR)   # best-face mode
        yunet_dets  = run_yunet(bgr)

        print(f"    OpenCV (mn=3)   : {len(opencv_dets)} det(s)")
        print(f"    C VJ (best_face): {len(c_dets) if c_dets is not None else 'N/A'}")
        print(f"    YuNet (0.5)     : {len(yunet_dets) if yunet_dets is not None else 'N/A'}")

        # Recall = "found at least 1 face" (binary per image, no GT labels needed)
        def recall(dets):
            if dets is None:
                return None
            return 1.0 if len(dets) >= 1 else 0.0

        row = {
            "name":       name,
            "size":       (w, h),
            "opencv_n":   len(opencv_dets),
            "c_n":        len(c_dets) if c_dets is not None else None,
            "yunet_n":    len(yunet_dets) if yunet_dets is not None else None,
            "opencv_rec": recall(opencv_dets),
            "c_rec":      recall(c_dets),
            "yunet_rec":  recall(yunet_dets),
        }
        print(f"    Recall: OpenCV={row['opencv_rec']}  "
              f"C_best={row['c_rec']}  YuNet={row['yunet_rec']}")

        # Latency
        cc = cv2.CascadeClassifier(CASCADE_XML)
        row["ocv_lat"] = _quick_py_latency(
            lambda: cc.detectMultiScale(gray, scaleFactor=SCALE_FACTOR,
                                        minNeighbors=MIN_NEIGHBORS, minSize=(24, 24)),
            N_LAT_RUNS,
        )

        if c_dets is not None and os.path.exists(BENCH_EXE):
            bench = run_c_bench(gray, SCALE_FACTOR, n_runs=N_LAT_RUNS, best_mode=True)
            if bench:
                row["c_lat"] = bench["mean_ms"]

        if yunet_dets is not None and ensure_yunet_model():
            det = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (w, h), 0.5, 0.3, 5000)
            row["yunet_lat"] = _quick_py_latency(lambda: det.detect(bgr), N_LAT_RUNS)

        all_rows.append(row)

    if not all_rows:
        return None

    _save_three_way_vis(all_rows)
    _make_tradeoff_scatter(all_rows)

    summary = {"per_image": all_rows, "n_images": len(all_rows)}
    return summary


def _save_three_way_vis(rows):
    ensure_outputs_dir()
    for r in rows:
        img_path = os.path.join(ROOT_DIR, "test", r["name"])
        bgr, gray = load_image(img_path)
        if gray is None:
            continue
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        opencv_dets = run_opencv(gray, SCALE_FACTOR, MIN_NEIGHBORS)
        c_dets      = run_c_best_face(gray, SCALE_FACTOR) or []
        yunet_dets  = run_yunet(bgr) or []

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"{r['name']}  |  "
            f"OpenCV minNeighbors=3 / C best_face / YuNet 0.5 conf",
            fontsize=11, fontweight="bold",
        )
        for ax, dets, title, col in [
            (axes[0], opencv_dets, f"OpenCV Haar (mn=3)\n{len(opencv_dets)} det(s)",   "#00CC44"),
            (axes[1], c_dets,      f"C VJ best_face\n{len(c_dets)} det(s)",            "#FF4422"),
            (axes[2], yunet_dets,  f"YuNet CNN\n{len(yunet_dets)} det(s)",             "#AA22FF"),
        ]:
            ax.imshow(img_rgb)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")
            for x, y, bw, bh in dets:
                ax.add_patch(patches.Rectangle(
                    (x, y), bw, bh, linewidth=2, edgecolor=col, facecolor="none"))

        plt.tight_layout()
        base = r["name"].rsplit(".", 1)[0]
        out = os.path.join(OUTPUTS_DIR, f"exp2_{base}_threeway.png")
        plt.savefig(out, dpi=100, bbox_inches="tight")
        plt.close()


def _make_tradeoff_scatter(rows):
    ensure_outputs_dir()
    fig, ax = plt.subplots(figsize=(10, 7))

    DETECTORS = [
        ("C VJ (best_face)", "c_lat",    "c_rec",    "o", "#2255CC"),
        ("OpenCV Haar",      "ocv_lat",  "opencv_rec","s", "#22AA44"),
        ("YuNet CNN",        "yunet_lat","yunet_rec", "^", "#CC4422"),
    ]

    for label, lat_k, rec_k, marker, color in DETECTORS:
        for r in rows:
            lat = r.get(lat_k)
            rec = r.get(rec_k)
            if lat is not None and rec is not None:
                ax.scatter(lat, rec, c=color, marker=marker, s=160, zorder=4,
                           label=label, edgecolors="black", linewidths=0.7)
                ax.annotate(r["name"].split(".")[0],
                            (lat, rec), textcoords="offset points",
                            xytext=(6, 4), fontsize=9, color=color)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=11,
              title="Detector", title_fontsize=10)

    ax.set_xlabel("Latency (ms, 20-run mean, I/O excluded)", fontsize=12)
    ax.set_ylabel("Recall (face found = 1, missed = 0)", fontsize=12)
    ax.set_title(
        "Exp 2: Accuracy vs. Latency Trade-off\n"
        "C: vj_detect_best_face()  |  upper-left = ideal",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(-0.1, 1.2)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "exp2_tradeoff_scatter.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Trade-off scatter saved: {out}")


if __name__ == "__main__":
    run_experiment()
