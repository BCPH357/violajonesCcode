#!/usr/bin/env python3
"""
Verify the C Viola-Jones detector against OpenCV.

Steps:
1. Download / use a test image (or create a synthetic one from the webcam snapshot)
2. Convert to grayscale, save as raw bytes
3. Run OpenCV CascadeClassifier (minNeighbors=0 for fair comparison)
4. Run our C test_vj.exe
5. Compare results
"""

import subprocess
import sys
import os
import struct
import urllib.request
import cv2
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)
CASCADE_XML = os.path.join(ROOT_DIR, "haarcascade_frontalface_default.xml")
TEST_EXE    = os.path.join(ROOT_DIR, "test", "test_vj.exe")
TMP_RAW     = os.path.join(ROOT_DIR, "test", "test_image.raw")
TMP_PNG     = os.path.join(ROOT_DIR, "test", "test_image.png")

# A small public-domain face image (Lena / Barbara replacements are tricky,
# use a simple synthetic test: white rectangle on grey background at known position)
# For a real test we download a sample face image.
SAMPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/320px-Gatto_europeo4.jpg"

def make_synthetic_image(w=160, h=120):
    """
    Create a synthetic grayscale image with a simple gradient.
    No real face, but useful for a smoke test (should detect 0 faces).
    """
    img = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            img[y, x] = (x + y) % 256
    return img


def run_opencv(gray_img, scale_factor, min_neighbors):
    cc = cv2.CascadeClassifier(CASCADE_XML)
    faces = cc.detectMultiScale(
        gray_img,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(24, 24),
    )
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def run_c_detector(raw_path, w, h, scale_factor, min_neighbors, best_only=False):
    mn_arg = "best" if best_only else str(min_neighbors)
    cmd = [TEST_EXE, raw_path, str(w), str(h),
           f"{scale_factor:.2f}", mn_arg]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("C detector error:", result.stderr)
        return []
    detections = []
    for line in result.stdout.splitlines():
        if line.strip().startswith("["):
            # Format: "  [0] x=10 y=20 w=30 h=30"
            parts = line.split()
            d = {}
            for p in parts:
                if "=" in p:
                    k, v = p.split("=")
                    d[k] = int(v)
            if "x" in d:
                detections.append((d["x"], d["y"], d["w"], d["h"]))
    return detections


def test_synthetic():
    print("=== Synthetic image smoke test ===")
    w, h = 160, 120
    img = make_synthetic_image(w, h)

    # Save raw
    with open(TMP_RAW, "wb") as f:
        f.write(img.tobytes())

    scale_factor = 1.2
    min_neighbors = 3

    opencv_dets = run_opencv(img, scale_factor, min_neighbors)
    c_dets = run_c_detector(TMP_RAW, w, h, scale_factor, min_neighbors)

    print(f"  OpenCV found: {len(opencv_dets)} face(s): {opencv_dets}")
    print(f"  C found:      {len(c_dets)} face(s): {c_dets}")
    print()


def test_with_real_image(img_path):
    print(f"=== Real image test: {img_path} ===")
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"  Cannot read {img_path}")
        return
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"  Image size: {w}x{h}")

    # Save raw
    with open(TMP_RAW, "wb") as f:
        f.write(gray.tobytes())
    cv2.imwrite(TMP_PNG, gray)

    for min_neighbors in [0, 3]:
        scale_factor = 1.2
        print(f"\n  --- minNeighbors={min_neighbors} ---")
        opencv_dets = run_opencv(gray, scale_factor, min_neighbors)
        c_dets = run_c_detector(TMP_RAW, w, h, scale_factor, min_neighbors)
        print(f"  OpenCV: {len(opencv_dets)} face(s): {opencv_dets}")
        print(f"  C:      {len(c_dets)} face(s): {c_dets}")

        if min_neighbors == 0:
            print(f"  (raw detections, counts may differ due to grouping differences)")
        else:
            ok = len(opencv_dets) == len(c_dets)
            print(f"  Match: {'YES' if ok else 'NO (count differs - check grouping)'}")
    print()


def download_sample():
    """Download a sample image for testing."""
    sample_path = os.path.join(ROOT_DIR, "test", "sample_face.jpg")
    if not os.path.exists(sample_path):
        print(f"Downloading sample image...")
        try:
            urllib.request.urlretrieve(SAMPLE_URL, sample_path)
        except Exception as e:
            print(f"  Download failed: {e}")
            return None
    return sample_path


def main():
    os.makedirs(os.path.join(ROOT_DIR, "test"), exist_ok=True)

    if not os.path.exists(TEST_EXE):
        print(f"ERROR: {TEST_EXE} not found. Run build.bat first.")
        sys.exit(1)

    # Smoke test with synthetic image
    test_synthetic()

    # Test with user-provided images
    extra_images = sys.argv[1:]
    if extra_images:
        for img_path in extra_images:
            test_with_real_image(img_path)
    else:
        # Try to find any JPG/PNG in the project
        for fname in os.listdir(ROOT_DIR):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                test_with_real_image(os.path.join(ROOT_DIR, fname))
                break
        else:
            print("No image found. Pass an image path as argument:")
            print(f"  python {__file__} photo.jpg")
            print()
            print("Hint: download a face photo and pass it here to compare with OpenCV.")

    print("Verification done.")


if __name__ == "__main__":
    main()
