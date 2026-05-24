"""
Experiment 1: C implementation vs OpenCV consistency.

Runs both detectors on every test image with the same parameters
and quantifies agreement via IoU-based matching.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import (
    OUTPUTS_DIR, VJ_EXE, ensure_outputs_dir,
    compute_iou, match_greedy,
    run_opencv, run_c_detect,
    get_test_images, load_image,
)

SCALE_FACTOR   = 1.2
MIN_NEIGHBORS  = 3
IOU_THRESH     = 0.5


def analyze_image(img_path):
    bgr, gray = load_image(img_path)
    if gray is None:
        return None
    opencv_dets = run_opencv(gray, SCALE_FACTOR, MIN_NEIGHBORS)
    c_dets      = run_c_detect(gray, SCALE_FACTOR, MIN_NEIGHBORS)
    if c_dets is None:
        return None

    h, w = gray.shape

    # Match C dets to OpenCV dets (OpenCV = reference)
    tp, fp, fn, ious = match_greedy(opencv_dets, c_dets, IOU_THRESH)

    # Coordinate deviations for matched pairs
    coord_devs = []
    ref_used = [False] * len(opencv_dets)
    for pred in c_dets:
        best_iou, best_j = 0.0, -1
        for j, ref in enumerate(opencv_dets):
            if ref_used[j]:
                continue
            iou = compute_iou(pred, ref)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= IOU_THRESH:
            ref = opencv_dets[best_j]
            ref_used[best_j] = True
            coord_devs.append({
                "dx": abs(pred[0] - ref[0]),
                "dy": abs(pred[1] - ref[1]),
                "dw": abs(pred[2] - ref[2]),
                "dh": abs(pred[3] - ref[3]),
            })

    return {
        "name":        os.path.basename(img_path),
        "size":        (w, h),
        "opencv_dets": opencv_dets,
        "c_dets":      c_dets,
        "tp": tp, "fp": fp, "fn": fn,
        "ious":        ious,
        "coord_devs":  coord_devs,
        "gray": gray,
    }


def save_side_by_side(r):
    ensure_outputs_dir()
    base = r["name"].rsplit(".", 1)[0]
    gray = r["gray"]
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"{r['name']}  |  scaleFactor={SCALE_FACTOR}  minNeighbors={MIN_NEIGHBORS}",
        fontsize=12, fontweight="bold",
    )
    for ax, dets, title, col in [
        (axes[0], r["opencv_dets"], "OpenCV CascadeClassifier", "#00CC44"),
        (axes[1], r["c_dets"],      "C Viola-Jones",            "#FF4422"),
    ]:
        ax.imshow(img_rgb)
        ax.set_title(f"{title}\n({len(dets)} det{'s' if len(dets)!=1 else ''})",
                     fontsize=11, fontweight="bold")
        ax.axis("off")
        for x, y, bw, bh in dets:
            ax.add_patch(patches.Rectangle(
                (x, y), bw, bh, linewidth=2, edgecolor=col, facecolor="none"))
            ax.text(x, y - 4, f"{bw}×{bh}", color=col, fontsize=7,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, f"exp1_{base}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def run_experiment():
    if not os.path.exists(VJ_EXE):
        print(f"ERROR: {VJ_EXE} not found. Run experiments/build_bench.bat first.")
        return None

    images = get_test_images()
    print(f"\n=== Experiment 1: C vs OpenCV Consistency ===")
    print(f"Images: {len(images)}  scaleFactor={SCALE_FACTOR}  minNeighbors={MIN_NEIGHBORS}\n")

    results = []
    for img_path in images:
        r = analyze_image(img_path)
        if r is None:
            print(f"  {os.path.basename(img_path)}: SKIP")
            continue
        mean_iou = np.mean(r["ious"]) if r["ious"] else float("nan")
        print(f"  {r['name']}: OpenCV={r['opencv_dets'].__len__()} "
              f"C={r['c_dets'].__len__()} "
              f"TP={r['tp']} FP={r['fp']} FN={r['fn']} "
              f"meanIoU={mean_iou:.3f}")
        save_side_by_side(r)
        results.append(r)

    if not results:
        return None

    # Aggregate
    all_ious = np.array([iou for r in results for iou in r["ious"]])
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    all_devs = [d for r in results for d in r["coord_devs"]]

    summary = {
        "n_images":    len(results),
        "total_opencv": sum(len(r["opencv_dets"]) for r in results),
        "total_c":      sum(len(r["c_dets"]) for r in results),
        "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
        "all_ious": all_ious,
        "mean_iou": float(all_ious.mean()) if len(all_ious) else float("nan"),
        "std_iou":  float(all_ious.std())  if len(all_ious) else float("nan"),
        "min_iou":  float(all_ious.min())  if len(all_ious) else float("nan"),
        "max_iou":  float(all_ious.max())  if len(all_ious) else float("nan"),
        "pct_iou_ge50": float((all_ious >= 0.5).mean()) if len(all_ious) else float("nan"),
        "coord_devs": all_devs,
        "per_image": results,
    }

    print(f"\n  Aggregate ({len(all_ious)} matched pairs):")
    print(f"  IoU  mean={summary['mean_iou']:.3f}  std={summary['std_iou']:.3f}  "
          f"min={summary['min_iou']:.3f}  max={summary['max_iou']:.3f}")
    print(f"  IoU>=0.5: {summary['pct_iou_ge50']*100:.1f}%  "
          f"Total: TP={total_tp} FP={total_fp} FN={total_fn}")

    if all_devs:
        dx = np.mean([d["dx"] for d in all_devs])
        dy = np.mean([d["dy"] for d in all_devs])
        dw = np.mean([d["dw"] for d in all_devs])
        dh = np.mean([d["dh"] for d in all_devs])
        print(f"  Mean coordinate δ: dx={dx:.1f} dy={dy:.1f} dw={dw:.1f} dh={dh:.1f} px")
        summary["mean_dx"] = dx
        summary["mean_dy"] = dy
        summary["mean_dw"] = dw
        summary["mean_dh"] = dh

    # IoU histogram
    if len(all_ious) > 0:
        ensure_outputs_dir()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_ious, bins=np.linspace(0, 1, 21),
                color="#4477CC", edgecolor="black", alpha=0.85)
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="IoU=0.5 thresh")
        ax.set_xlabel("IoU (C vs OpenCV matched pairs)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Exp 1: IoU Distribution — C Detector vs OpenCV",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.set_xlim(0, 1)
        plt.tight_layout()
        hist_path = os.path.join(OUTPUTS_DIR, "exp1_iou_histogram.png")
        plt.savefig(hist_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: {hist_path}")

    return summary


if __name__ == "__main__":
    run_experiment()
