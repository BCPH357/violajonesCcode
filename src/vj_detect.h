#ifndef VJ_DETECT_H
#define VJ_DETECT_H

#include "vj_types.h"
#include <stdint.h>

/*
 * Detect faces using the Viola-Jones cascade classifier.
 *
 * Parameters:
 *   cascade      - Pointer to the cascade data structure.
 *   gray_img     - Grayscale image (row-major, 8-bit).
 *   img_w, img_h - Image dimensions.
 *   scale_factor - Scale multiplier per iteration (e.g. 1.2).
 *   min_neighbors- Minimum overlap count to keep a detection.
 *                  0 = no grouping (raw detections).
 *   det_buf      - Caller-allocated output buffer for detections.
 *   max_det      - Capacity of det_buf.
 *   work_buf     - Working memory for integral images.
 *   work_buf_size- Size in bytes.  Must be >= vj_work_buf_size(img_w, img_h).
 *
 * Returns the number of detections written to det_buf.
 */
int vj_detect_faces(const vj_cascade_t *cascade,
                    const uint8_t *gray_img, int img_w, int img_h,
                    float scale_factor,
                    int min_neighbors,
                    vj_detection_t *det_buf, int max_det,
                    void *work_buf, int work_buf_size);

/*
 * Minimum working buffer size required for an image of dimensions w x h.
 * Accounts for both integral image (uint32) and squared integral image (uint64).
 */
static inline int vj_work_buf_size(int w, int h)
{
    int padded = (w + 1) * (h + 1);
    return padded * (int)(sizeof(uint32_t) + sizeof(uint64_t));
}

/*
 * Group overlapping rectangles (non-maximum suppression).
 * Merges detections in-place, returns the new count.
 *
 * If scores_out != NULL, it must point to an array of at least `count` ints.
 * After grouping, scores_out[i] holds the vote count (number of raw detections
 * that merged into the i-th surviving group) — useful for picking the best face.
 */
int vj_group_rectangles(vj_detection_t *dets, int count, int min_neighbors,
                        int *scores_out);

/*
 * Detect the single highest-confidence face in the image.
 *
 * "Highest confidence" = the group with the most overlapping raw detections
 * (most windows agreed it was a face).
 *
 * Returns 1 if a face was found and writes the result to *best.
 * Returns 0 if no face was found.
 *
 * scale_factor : e.g. 1.2
 * work_buf     : same buffer used by vj_detect_faces
 * work_buf_size: must be >= vj_work_buf_size(img_w, img_h)
 *
 * Internally uses a stack buffer capped at VJ_BEST_FACE_MAX_RAW raw detections.
 */
#define VJ_BEST_FACE_MAX_RAW 2048

int vj_detect_best_face(const vj_cascade_t *cascade,
                        const uint8_t *gray_img, int img_w, int img_h,
                        float scale_factor,
                        vj_detection_t *best,
                        void *work_buf, int work_buf_size);

#endif /* VJ_DETECT_H */
