"""
Experiment 3: Cascade Intrinsic Metrics + Latency.

Headline: cascade computational structure (per-stage rejection rate,
window candidates, average features evaluated per window, no-multiply property).

Absolute latency is secondary context, explicitly labelled:
  - C VJ : unoptimized scalar C, clock() CPU time — reference implementation only
  - OpenCV: same algorithm, SIMD-optimized, perf_counter wall clock
  - YuNet : MobileNet-based SSD CNN, perf_counter wall clock

This latency comparison does NOT reflect algorithmic computational complexity;
the same VJ algorithm in OpenCV is ~14x faster solely due to SIMD.
Hardware efficiency (no-multiply, low area, low power) will be validated in
the FPGA/ASIC phase — PC latency is not the research claim.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    OUTPUTS_DIR, BENCH_EXE, ensure_outputs_dir,
    run_c_bench, time_opencv, time_yunet,
    get_test_images, load_image,
    compute_window_count,
)

SCALE_FACTOR  = 1.2
MIN_NEIGHBORS = 3
N_RUNS        = 50


def _extract_stage_stats(bench_dict):
    """Extract per-stage stats list from bench_vj output dict."""
    if bench_dict is None:
        return []
    ns = int(bench_dict.get("num_stages", 0))
    stages = []
    for s in range(ns):
        entered  = bench_dict.get(f"stage_{s}_entered")
        rejected = bench_dict.get(f"stage_{s}_rejected")
        rej_pct  = bench_dict.get(f"stage_{s}_rej_pct")
        num_weak = bench_dict.get(f"stage_{s}_num_weak")
        if entered is None:
            break
        stages.append({
            "stage":    s,
            "entered":  int(entered),
            "rejected": int(rejected),
            "rej_pct":  float(rej_pct),
            "num_weak": int(num_weak),
        })
    return stages


def run_experiment():
    print(f"\n=== Experiment 3: Cascade Intrinsic Metrics + Latency ===")
    print(f"N_runs={N_RUNS}  scaleFactor={SCALE_FACTOR}  minNeighbors={MIN_NEIGHBORS}")
    print(f"[C = unoptimized scalar reference; OpenCV/YuNet = SIMD-optimized]")

    c_available = os.path.exists(BENCH_EXE)
    if not c_available:
        print(f"WARNING: {BENCH_EXE} not found -- C timing/stats skipped.")

    images = get_test_images()
    all_rows = []

    for img_path in images:
        bgr, gray = load_image(img_path)
        if gray is None:
            continue
        name = os.path.basename(img_path)
        h, w = gray.shape
        win_count = compute_window_count(w, h)
        print(f"\n  {name} ({w}x{h}, {win_count:,} window candidates):")

        row = {"name": name, "size": (w, h), "win_count": win_count}

        # C VJ (timing + cascade intrinsic stats)
        if c_available:
            bench = run_c_bench(gray, SCALE_FACTOR, MIN_NEIGHBORS, N_RUNS)
            if bench:
                max_feats = int(bench.get("max_feats_per_window", 0))
                avg_feats = float(bench.get("avg_feats_per_window", 0))
                eff_pct   = (1.0 - avg_feats / max_feats) * 100.0 if max_feats else 0.0
                row.update({
                    "c_mean":       bench["mean_ms"],
                    "c_std":        bench["std_ms"],
                    "c_min":        bench["min_ms"],
                    "c_max":        bench["max_ms"],
                    "c_wins":       int(bench.get("window_candidates", 0)),
                    "max_feats":    max_feats,
                    "total_feats":  int(bench.get("total_feats_evaluated", 0)),
                    "avg_feats":    avg_feats,
                    "eff_pct":      eff_pct,
                    "stage_stats":  _extract_stage_stats(bench),
                    "num_stages":   int(bench.get("num_stages", 0)),
                })
                print(f"    [Intrinsic] avg_feats/window: {avg_feats:.1f} / {max_feats}"
                      f"  ({eff_pct:.1f}% skipped by cascade early-exit)")
                print(f"    [Intrinsic] NO MULTIPLICATIONS in feature eval path "
                      f"(integral image = additions only)")
                # Print stage 0 and cumulative rejection at key stages
                stages = row["stage_stats"]
                if stages:
                    total = stages[0]["entered"]
                    cum_survived = total
                    for st in stages:
                        cum_survived -= st["rejected"]
                        if st["stage"] in (0, 1, 2, 4, 9, 24):
                            print(f"      Stage {st['stage']:2d} ({st['num_weak']:3d} feat): "
                                  f"rej {st['rej_pct']:5.1f}%  "
                                  f"surviving: {cum_survived:,}/{total:,}")
                print(f"    [Latency]   C VJ (scalar): "
                      f"{row['c_mean']:.1f} +/- {row['c_std']:.1f} ms")

        # OpenCV Haar
        ocv = time_opencv(gray, N_RUNS, SCALE_FACTOR, MIN_NEIGHBORS)
        row.update({"ocv_mean": ocv["mean_ms"], "ocv_std": ocv["std_ms"],
                    "ocv_min": ocv["min_ms"],   "ocv_max": ocv["max_ms"]})
        print(f"    [Latency]   OpenCV (SIMD):  "
              f"{row['ocv_mean']:.1f} +/- {row['ocv_std']:.1f} ms")

        # YuNet CNN
        yunet = time_yunet(bgr, N_RUNS)
        if yunet:
            row.update({"yu_mean": yunet["mean_ms"], "yu_std": yunet["std_ms"],
                        "yu_min": yunet["min_ms"],   "yu_max": yunet["max_ms"]})
            print(f"    [Latency]   YuNet (CNN):   "
                  f"{row['yu_mean']:.1f} +/- {row['yu_std']:.1f} ms")
        else:
            print(f"    [Latency]   YuNet: UNAVAILABLE")

        all_rows.append(row)

    if not all_rows:
        return None

    def agg(key):
        vals = [r[key] for r in all_rows if key in r]
        return float(np.mean(vals)) if vals else None

    summary = {
        "per_image":    all_rows,
        "n_runs":       N_RUNS,
        "c_mean_avg":   agg("c_mean"),
        "ocv_mean_avg": agg("ocv_mean"),
        "yu_mean_avg":  agg("yu_mean"),
        "avg_feats_avg": agg("avg_feats"),
        "max_feats":    next((r["max_feats"] for r in all_rows if "max_feats" in r), None),
        "num_stages":   next((r["num_stages"] for r in all_rows if "num_stages" in r), None),
    }

    print("\n  Aggregate mean latency (unoptimized C scalar / SIMD OpenCV / SIMD YuNet):")
    if summary["c_mean_avg"] is not None:
        print(f"    C VJ (scalar, ref): {summary['c_mean_avg']:.1f} ms")
    print(f"    OpenCV (SIMD):      {summary['ocv_mean_avg']:.1f} ms")
    if summary["yu_mean_avg"] is not None:
        print(f"    YuNet (SIMD/CNN):   {summary['yu_mean_avg']:.1f} ms")

    ensure_outputs_dir()
    _make_bar_chart(all_rows, summary)
    _make_cascade_chart(all_rows)

    return summary


def _make_bar_chart(rows, summary):
    labels = [r["name"] for r in rows]
    x = np.arange(len(labels))
    width = 0.22

    groups = []
    has_c     = any("c_mean" in r for r in rows)
    has_yunet = any("yu_mean" in r for r in rows)

    if has_c:
        groups.append(("C VJ (scalar, ref.)", "c_mean",  "c_std",  "#2255CC"))
    groups.append(    ("OpenCV Haar (SIMD)", "ocv_mean", "ocv_std", "#22AA44"))
    if has_yunet:
        groups.append(("YuNet CNN (SIMD)",   "yu_mean",  "yu_std",  "#CC4422"))

    n = len(groups)
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

    fig, ax = plt.subplots(figsize=(10, 6))
    for (label, mk, sk, col), off in zip(groups, offsets):
        means = [r.get(mk, 0) for r in rows]
        stds  = [r.get(sk, 0) for r in rows]
        ax.bar(x + off, means, width * 0.90, label=label, color=col,
               yerr=stds, capsize=4, alpha=0.85,
               error_kw={"elinewidth": 1.5, "ecolor": "black"})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_xlabel("Test Image", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(
        "Exp 3: Face Detector Latency (secondary context)\n"
        f"mean +/- std, N={summary['n_runs']} runs, I/O excluded\n"
        "C = unoptimized scalar; OpenCV/YuNet = SIMD-optimized — not an apples-to-apples comparison",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "exp3_latency.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  Latency chart saved: {out}")


def _make_cascade_chart(rows):
    """Per-stage rejection rate for each test image (headline chart for Exp3)."""
    rows_with_stages = [r for r in rows if r.get("stage_stats")]
    if not rows_with_stages:
        return

    colors = ["#2255CC", "#22AA44", "#CC4422", "#AA22FF"]
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, r in enumerate(rows_with_stages):
        stages = r["stage_stats"]
        stage_ids = [s["stage"] for s in stages]
        rej_pcts  = [s["rej_pct"] for s in stages]
        ax.plot(stage_ids, rej_pcts,
                marker="o", markersize=5, linewidth=2,
                color=colors[i % len(colors)],
                label=r["name"])

    ax.set_xlabel("Cascade Stage Index", fontsize=12)
    ax.set_ylabel("Windows Rejected at This Stage (%)", fontsize=12)
    ax.set_title(
        "Exp 3: Cascade Early-Rejection Profile (headline intrinsic metric)\n"
        "High early-stage rejection => most windows exit after Stage 0-4 "
        "(avg ~30 features vs 2913 max)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=11, title="Image")
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "exp3_cascade_rejection.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Cascade rejection chart saved: {out}")


if __name__ == "__main__":
    run_experiment()
