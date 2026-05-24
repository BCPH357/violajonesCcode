#!/usr/bin/env python3
"""
Convert OpenCV Haar cascade XML to C arrays for MCU deployment.

Usage:
    python convert_cascade.py haarcascade_frontalface_default.xml --output-dir ../src/
"""

import argparse
import os
import xml.etree.ElementTree as ET


def parse_cascade(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    cascade = root.find("cascade")
    if cascade is None:
        raise ValueError("No <cascade> element found in XML")

    window_w = int(cascade.find("width").text)
    window_h = int(cascade.find("height").text)

    # --- Parse features ---
    features = []
    features_el = cascade.find("features")
    for feat_el in features_el:
        rects_el = feat_el.find("rects")
        rects = []
        for r in rects_el:
            parts = r.text.strip().split()
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            weight = float(parts[4].rstrip("."))
            rects.append((x, y, w, h, weight))
        features.append(rects)

    # --- Parse stages and weak classifiers ---
    stages = []
    weak_classifiers = []
    stages_el = cascade.find("stages")
    weak_idx = 0

    for stage_el in stages_el:
        num_weak = int(stage_el.find("maxWeakCount").text)
        stage_threshold = float(stage_el.find("stageThreshold").text)
        stages.append({
            "num_weak": num_weak,
            "weak_start_idx": weak_idx,
            "stage_threshold": stage_threshold,
        })

        wc_el = stage_el.find("weakClassifiers")
        for wc in wc_el:
            internal = wc.find("internalNodes").text.strip().split()
            leaves = wc.find("leafValues").text.strip().split()

            # internalNodes format: "0 -1 <feature_idx> <threshold>"
            feature_idx = int(internal[2])
            threshold = float(internal[3])
            left_val = float(leaves[0])
            right_val = float(leaves[1])

            weak_classifiers.append({
                "feature_idx": feature_idx,
                "threshold": threshold,
                "left_val": left_val,
                "right_val": right_val,
            })
            weak_idx += 1

    return {
        "window_w": window_w,
        "window_h": window_h,
        "features": features,
        "stages": stages,
        "weak_classifiers": weak_classifiers,
    }


def float_to_c(val):
    return f"{val:.17e}f"


def generate_c_source(cascade_data):
    lines = []
    lines.append('#include "vj_cascade_data.h"')
    lines.append("")

    features = cascade_data["features"]
    stages = cascade_data["stages"]
    weaks = cascade_data["weak_classifiers"]

    # --- Features array ---
    lines.append(f"const vj_feature_t vj_features[{len(features)}] = {{")
    for i, rects in enumerate(features):
        num_rects = len(rects)
        rect_strs = []
        for r in rects:
            rect_strs.append(f"{{{r[0]}, {r[1]}, {r[2]}, {r[3]}, {float_to_c(r[4])}}}")
        # Pad to 3 rects
        while len(rect_strs) < 3:
            rect_strs.append("{0, 0, 0, 0, 0.0f}")
        comma = "," if i < len(features) - 1 else ""
        lines.append(f"    {{{{{', '.join(rect_strs)}}}, {num_rects}}}{comma}")
    lines.append("};")
    lines.append("")

    # --- Weak classifiers array ---
    lines.append(f"const vj_weak_classifier_t vj_weak_classifiers[{len(weaks)}] = {{")
    for i, wc in enumerate(weaks):
        comma = "," if i < len(weaks) - 1 else ""
        lines.append(
            f"    {{{wc['feature_idx']}, "
            f"{float_to_c(wc['threshold'])}, "
            f"{float_to_c(wc['left_val'])}, "
            f"{float_to_c(wc['right_val'])}}}{comma}"
        )
    lines.append("};")
    lines.append("")

    # --- Stages array ---
    lines.append(f"const vj_stage_t vj_stages[{len(stages)}] = {{")
    for i, s in enumerate(stages):
        comma = "," if i < len(stages) - 1 else ""
        lines.append(
            f"    {{{s['num_weak']}, {s['weak_start_idx']}, "
            f"{float_to_c(s['stage_threshold'])}}}{comma}"
        )
    lines.append("};")
    lines.append("")

    # --- Cascade struct ---
    lines.append("const vj_cascade_t vj_cascade = {")
    lines.append(f"    .window_w = {cascade_data['window_w']},")
    lines.append(f"    .window_h = {cascade_data['window_h']},")
    lines.append(f"    .num_stages = {len(stages)},")
    lines.append(f"    .num_features = {len(features)},")
    lines.append(f"    .num_weak_total = {len(weaks)},")
    lines.append("    .stages = vj_stages,")
    lines.append("    .weak_classifiers = vj_weak_classifiers,")
    lines.append("    .features = vj_features")
    lines.append("};")
    lines.append("")

    return "\n".join(lines)


def generate_c_header(cascade_data):
    lines = []
    lines.append("#ifndef VJ_CASCADE_DATA_H")
    lines.append("#define VJ_CASCADE_DATA_H")
    lines.append("")
    lines.append('#include "vj_types.h"')
    lines.append("")
    lines.append(f"#define VJ_NUM_FEATURES {len(cascade_data['features'])}")
    lines.append(f"#define VJ_NUM_STAGES {len(cascade_data['stages'])}")
    lines.append(f"#define VJ_NUM_WEAK_CLASSIFIERS {len(cascade_data['weak_classifiers'])}")
    lines.append(f"#define VJ_WINDOW_W {cascade_data['window_w']}")
    lines.append(f"#define VJ_WINDOW_H {cascade_data['window_h']}")
    lines.append("")
    lines.append(f"extern const vj_feature_t vj_features[{len(cascade_data['features'])}];")
    lines.append(f"extern const vj_weak_classifier_t vj_weak_classifiers[{len(cascade_data['weak_classifiers'])}];")
    lines.append(f"extern const vj_stage_t vj_stages[{len(cascade_data['stages'])}];")
    lines.append("extern const vj_cascade_t vj_cascade;")
    lines.append("")
    lines.append("#endif /* VJ_CASCADE_DATA_H */")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert OpenCV Haar cascade XML to C arrays")
    parser.add_argument("xml_file", help="Path to haarcascade XML file")
    parser.add_argument("--output-dir", default=".", help="Output directory for generated C files")
    args = parser.parse_args()

    print(f"Parsing {args.xml_file} ...")
    data = parse_cascade(args.xml_file)

    print(f"  Window:            {data['window_w']}x{data['window_h']}")
    print(f"  Features:          {len(data['features'])}")
    print(f"  Stages:            {len(data['stages'])}")
    print(f"  Weak classifiers:  {len(data['weak_classifiers'])}")

    os.makedirs(args.output_dir, exist_ok=True)

    c_path = os.path.join(args.output_dir, "vj_cascade_data.c")
    h_path = os.path.join(args.output_dir, "vj_cascade_data.h")

    c_src = generate_c_source(data)
    h_src = generate_c_header(data)

    with open(c_path, "w", newline="\n") as f:
        f.write(c_src)
    print(f"  Generated {c_path} ({len(c_src)} bytes)")

    with open(h_path, "w", newline="\n") as f:
        f.write(h_src)
    print(f"  Generated {h_path} ({len(h_src)} bytes)")

    print("Done.")


if __name__ == "__main__":
    main()
