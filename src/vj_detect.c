#include "vj_detect.h"
#include "vj_integral.h"
#include <math.h>
#include <string.h>

/* ---------- single-window cascade evaluation ---------- */

static int vj_evaluate_window(const vj_cascade_t *cascade,
                              const uint32_t *ii, const uint64_t *sii,
                              int ii_stride,
                              int win_x, int win_y,
                              int win_w, int win_h)
{
    float inv_area = 1.0f / (float)(win_w * win_h);

    /* mean and variance via integral images */
    uint32_t sum  = vj_rect_sum(ii,  ii_stride, win_x, win_y, win_w, win_h);
    uint64_t sqsum = vj_rect_sum_sq(sii, ii_stride, win_x, win_y, win_w, win_h);

    float mean     = (float)sum * inv_area;
    float variance = (float)sqsum * inv_area - mean * mean;
    float stdev    = (variance > 0.0f) ? sqrtf(variance) : 1.0f;

    float scale_x = (float)win_w / (float)cascade->window_w;
    float scale_y = (float)win_h / (float)cascade->window_h;

    for (int s = 0; s < cascade->num_stages; s++) {
        const vj_stage_t *stage = &cascade->stages[s];
        float stage_sum = 0.0f;

        for (int w = 0; w < stage->num_weak; w++) {
            const vj_weak_classifier_t *wc =
                &cascade->weak_classifiers[stage->weak_start_idx + w];
            const vj_feature_t *feat = &cascade->features[wc->feature_idx];

            /* evaluate feature: weighted sum of rectangle regions */
            float feat_val = 0.0f;
            for (int r = 0; r < feat->num_rects; r++) {
                const vj_rect_t *rc = &feat->rects[r];
                int rx = win_x + (int)(rc->x * scale_x);
                int ry = win_y + (int)(rc->y * scale_y);
                int rw = (int)(rc->w * scale_x);
                int rh = (int)(rc->h * scale_y);
                if (rw < 1) rw = 1;
                if (rh < 1) rh = 1;

                float rect_sum = (float)vj_rect_sum(ii, ii_stride, rx, ry, rw, rh);
                feat_val += rect_sum * rc->weight;
            }

            /* normalize by window area and standard deviation */
            feat_val *= inv_area;

            if (feat_val < wc->threshold * stdev)
                stage_sum += wc->left_val;
            else
                stage_sum += wc->right_val;
        }

        if (stage_sum < stage->stage_threshold)
            return 0; /* rejected */
    }
    return 1; /* passed all stages */
}

/* ---------- multi-scale sliding window ---------- */

int vj_detect_faces(const vj_cascade_t *cascade,
                    const uint8_t *gray_img, int img_w, int img_h,
                    float scale_factor,
                    int min_neighbors,
                    vj_detection_t *det_buf, int max_det,
                    void *work_buf, int work_buf_size)
{
    int required = vj_work_buf_size(img_w, img_h);
    if (work_buf_size < required)
        return -1;

    int padded = (img_w + 1) * (img_h + 1);
    uint32_t *ii  = (uint32_t *)work_buf;
    uint64_t *sii = (uint64_t *)((uint8_t *)work_buf + padded * sizeof(uint32_t));

    vj_compute_integral_images(gray_img, img_w, img_h, ii, sii);

    int ii_stride = img_w + 1;
    int det_count = 0;
    float scale = 1.0f;

    while (1) {
        /* Round to nearest (matches OpenCV cvRound) rather than floor-truncate.
         * At scale^4≈2.07: floor gives 49 px, round gives 50 px — 1 px affects
         * every feature offset (scale_x = win_w/24), hence cascade pass rate. */
        int win_w = (int)(cascade->window_w * scale + 0.5f);
        int win_h = (int)(cascade->window_h * scale + 0.5f);
        if (win_w > img_w || win_h > img_h)
            break;

        /* Match OpenCV's yStep rule:
         *   scale <= 2: yStep=2 in downsampled coords → 2*scale px in original
         *   scale >  2: yStep=1 in downsampled coords →   scale px in original
         * Previously always used 2*scale, undersampling large windows by 2×. */
        int ystep = (scale > 2.0f) ? 1 : 2;
        int step  = (int)(ystep * scale + 0.5f);
        if (step < 1) step = 1;

        for (int y = 0; y + win_h <= img_h; y += step) {
            for (int x = 0; x + win_w <= img_w; x += step) {
                if (vj_evaluate_window(cascade, ii, sii, ii_stride,
                                       x, y, win_w, win_h)) {
                    if (det_count < max_det) {
                        det_buf[det_count].x = (int16_t)x;
                        det_buf[det_count].y = (int16_t)y;
                        det_buf[det_count].w = (int16_t)win_w;
                        det_buf[det_count].h = (int16_t)win_h;
                        det_count++;
                    }
                }
            }
        }

        scale *= scale_factor;
    }

    if (min_neighbors > 0 && det_count > 0)
        det_count = vj_group_rectangles(det_buf, det_count, min_neighbors, NULL);

    return det_count;
}

/* ---------- rectangle grouping (NMS) ---------- */

/*
 * Simple disjoint-set (union-find) using the work stack on the caller side.
 * We keep labels[] on the stack with max_det entries.
 */

static int uf_find(int *labels, int i)
{
    int root = i;
    while (labels[root] != root) root = labels[root];
    while (labels[i] != root) { int next = labels[i]; labels[i] = root; i = next; }
    return root;
}

static void uf_union(int *labels, int a, int b)
{
    a = uf_find(labels, a);
    b = uf_find(labels, b);
    if (a != b) labels[b] = a;
}

/*
 * Core grouping: build labels[] via union-find using the OpenCV criterion.
 * labels must be pre-allocated with `count` elements.
 * Returns the number of elements actually processed (capped at VJ_BEST_FACE_MAX_RAW).
 *
 * OpenCV eps=0.2 criterion: two rects r1, r2 merge when ALL four hold:
 *   |x1-x2|           <= 0.2 * min(w1,w2)
 *   |y1-y2|           <= 0.2 * min(w1,w2)
 *   |(x1+w1)-(x2+w2)| <= 0.2 * min(w1,w2)   ← rejects very different scales
 *   |(y1+h1)-(y2+h2)| <= 0.2 * min(w1,w2)
 * Integer arithmetic: multiply differences by 5 and compare against min_w.
 */
static int build_labels(const vj_detection_t *dets, int count, int *labels)
{
    if (count > VJ_BEST_FACE_MAX_RAW) count = VJ_BEST_FACE_MAX_RAW;
    for (int i = 0; i < count; i++) labels[i] = i;

    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            int x1 = dets[i].x, y1 = dets[i].y, w1 = dets[i].w, h1 = dets[i].h;
            int x2 = dets[j].x, y2 = dets[j].y, w2 = dets[j].w, h2 = dets[j].h;

            /* OpenCV's exact groupRectangles predicate (eps = 0.2):
             *   delta = 0.2 * (min(w1,h1) + min(w2,h2)) / 2
             * Using AVERAGE of both min-dims, not just min(w1,w2).
             * This allows adjacent scale-level windows to merge:
             *   e.g. 148 & 178: (148+178)/2*0.2 = 32.6 >= 30 → merged
             * Integer form: multiply diffs by 10, compare to (md1+md2). */
            int md1 = w1 < h1 ? w1 : h1;
            int md2 = w2 < h2 ? w2 : h2;
            int ref = md1 + md2;  /* = 2 * avg_min_dim */

            if (abs(x1 - x2)            * 10 <= ref &&
                abs(y1 - y2)            * 10 <= ref &&
                abs((x1+w1) - (x2+w2)) * 10 <= ref &&
                abs((y1+h1) - (y2+h2)) * 10 <= ref)
            {
                uf_union(labels, i, j);
            }
        }
    }
    for (int i = 0; i < count; i++) labels[i] = uf_find(labels, i);
    return count;
}

int vj_group_rectangles(vj_detection_t *dets, int count, int min_neighbors,
                        int *scores_out)
{
    if (count <= 1) {
        if (scores_out && count == 1) scores_out[0] = 1;
        return count;
    }

    int labels[VJ_BEST_FACE_MAX_RAW];
    count = build_labels(dets, count, labels);

    int out = 0;
    for (int i = 0; i < count; i++) {
        if (labels[i] != i) continue; /* not a root */

        int64_t sx = 0, sy = 0, sw = 0, sh = 0;
        int n = 0;
        for (int j = 0; j < count; j++) {
            if (labels[j] == i) {
                sx += dets[j].x; sy += dets[j].y;
                sw += dets[j].w; sh += dets[j].h;
                n++;
            }
        }

        if (n >= min_neighbors) {
            dets[out].x = (int16_t)(sx / n);
            dets[out].y = (int16_t)(sy / n);
            dets[out].w = (int16_t)(sw / n);
            dets[out].h = (int16_t)(sh / n);
            if (scores_out) scores_out[out] = n;
            out++;
        }
    }
    return out;
}

/* ---------- single best face ---------- */

int vj_detect_best_face(const vj_cascade_t *cascade,
                        const uint8_t *gray_img, int img_w, int img_h,
                        float scale_factor,
                        vj_detection_t *best,
                        void *work_buf, int work_buf_size)
{
    vj_detection_t raw_buf[VJ_BEST_FACE_MAX_RAW];
    int labels[VJ_BEST_FACE_MAX_RAW];

    /* 1. Collect all raw detections */
    int n = vj_detect_faces(cascade, gray_img, img_w, img_h,
                            scale_factor, 0,
                            raw_buf, VJ_BEST_FACE_MAX_RAW,
                            work_buf, work_buf_size);
    if (n <= 0) return 0;

    /* 2. Build group labels */
    n = build_labels(raw_buf, n, labels);

    /* 3. Count votes per root and find the best group */
    int best_root  = -1;
    int best_score = 0;
    for (int i = 0; i < n; i++) {
        if (labels[i] != i) continue;
        int votes = 0;
        for (int j = 0; j < n; j++)
            if (labels[j] == i) votes++;
        if (votes > best_score) { best_score = votes; best_root = i; }
    }
    if (best_root < 0) return 0;

    /* 4. Compute a robust representative for the best group.
     *
     * Strategy: find the mode window-size (most detections at one scale),
     * then include all group members within ONE scale-step of the mode
     * (i.e., w in [mode/scale_factor, mode*scale_factor]).
     * This replicates OpenCV's behaviour of averaging adjacent scale levels
     * while excluding faraway outliers from unrelated false-positives.
     *
     * Use integer approximation: one scale step of 1.2 means the adjacent
     * window is within ±20% of the current.  We include w s.t.
     *   mode * 5 <= w * 6   (w >= mode/1.2)
     *   w * 5    <= mode * 6 (w <= mode*1.2)
     */
    int mode_w = 0, mode_cnt = 0;
    for (int i = 0; i < n; i++) {
        if (labels[i] != best_root) continue;
        int w = raw_buf[i].w;
        int cnt = 0;
        for (int j = 0; j < n; j++)
            if (labels[j] == best_root && raw_buf[j].w == w) cnt++;
        if (cnt > mode_cnt) { mode_cnt = cnt; mode_w = w; }
    }

    /* Include members within ±one scale step of mode */
    int64_t sx = 0, sy = 0, sw = 0, sh = 0;
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        if (labels[i] != best_root) continue;
        int w = raw_buf[i].w;
        /* w >= mode/1.2  →  mode*5 <= w*6
         * w <= mode*1.2  →  w*5    <= mode*6  */
        if (mode_w * 5 <= w * 6 && w * 5 <= mode_w * 6) {
            sx += raw_buf[i].x;
            sy += raw_buf[i].y;
            sw += raw_buf[i].w;
            sh += raw_buf[i].h;
            cnt++;
        }
    }

    best->x = (int16_t)(sx / cnt);
    best->y = (int16_t)(sy / cnt);
    best->w = (int16_t)(sw / cnt);
    best->h = (int16_t)(sh / cnt);
    return 1;
}
