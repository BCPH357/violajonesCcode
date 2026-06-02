#ifndef VJ_FIXED_H
#define VJ_FIXED_H

#include "vj_types.h"
#include <stdint.h>

/*
 * Fixed-point framework for vj_evaluate_window_fixed (Steps 1-4 complete).
 *
 * HLS-ready evaluator: fully integer, zero floating-point in hot path.
 *   - No inv_area division  (Step 1: absorbed, both sides ×area)
 *   - No weight multiply    (Step 2: shift/negate via VJ_WCODE_*)
 *   - No coord float mul    (Step 3: precomputed vj_scaled_feature_t lookup)
 *   - No sqrtf              (Step 4: four-branch __int128 square comparison)
 *
 * Q formats (from docs/QUANT_ANALYSIS.md):
 *   threshold        Q1.15  int32_t   range ±1,    precision ~3e-5
 *   left/right_val   Q6.10  int32_t   range ±63,   covers [-30,+10]
 *   stage_sum        Q6.10  int32_t   same format, direct comparison
 *   stage_threshold  Q6.10  int32_t   same format
 *
 * Dependency closure for HLS synthesis:
 *   vj_types.h                  — vj_cascade_t, VJ_MAX_RECTS
 *   vj_integral.h               — vj_rect_sum() / vj_rect_sum_sq() (inline)
 *   vj_fixed.h + vj_fixed.c    — this file
 *   stdlib.h (malloc/free)      — build/free helpers only; not in evaluator
 *
 * Entry point for HLS: vj_evaluate_window_fixed()
 *   Call graph (all inlined / static):
 *     vj_rect_sum()       — 4 array reads, 3 adds
 *     vj_rect_sum_sq()    — same pattern on squared-sum II
 *     vj_apply_weight()   — switch on 2-bit weight_code
 *     cmp_lhs_lt_rhs()    — sign-split uint64 square compare (no wide_t)
 */

/* ---------- Q-format constants ---------- */
#define VJ_Q_THRESH  15   /* threshold:      Q1.15 */
#define VJ_Q_LRVAL   10   /* left/right_val: Q6.10 */

/* ---------- weight codes (Step 2) ----------
 * All Haar rect weights are exactly {-1, 2, 3} (confirmed in QUANT_ANALYSIS.md).
 * Encode as a 2-bit code; apply_weight() converts to shift/negate. */
#define VJ_WCODE_NEG1  0   /* weight = -1  →  -rs          */
#define VJ_WCODE_POS2  1   /* weight = +2  →  rs << 1      */
#define VJ_WCODE_POS3  2   /* weight = +3  →  (rs<<1) + rs */

static inline int32_t vj_apply_weight(int32_t rs, int8_t code)
{
    switch (code) {
        case VJ_WCODE_NEG1: return -rs;
        case VJ_WCODE_POS2: return rs << 1;
        case VJ_WCODE_POS3: return (rs << 1) + rs;
        default:            return 0;   /* unreachable */
    }
}

/* ---------- fixed-point rect (Step 2: weight → int8 code) ---------- */
typedef struct {
    int16_t x, y, w, h;
    int8_t  weight_code;   /* VJ_WCODE_* */
} vj_rect_fixed_t;

typedef struct {
    vj_rect_fixed_t rects[VJ_MAX_RECTS];
    uint8_t         num_rects;
} vj_feature_fixed_t;

/* ---------- fixed-point weak classifier ---------- */
typedef struct {
    uint16_t feature_idx;          /* index into features_fixed[] */
    int32_t  threshold_q15;        /* round(threshold  * (1<<15)) */
    int32_t  left_val_q10;         /* round(left_val   * (1<<10)) */
    int32_t  right_val_q10;        /* round(right_val  * (1<<10)) */
} vj_wc_fixed_t;

/* ---------- fixed-point stage ---------- */
typedef struct {
    uint16_t num_weak;
    uint16_t weak_start_idx;
    int32_t  stage_threshold_q10;  /* round(stage_threshold * (1<<10)) */
} vj_stage_fixed_t;

/* ---------- fixed-point cascade ---------- */
typedef struct {
    int                      num_stages;
    int                      num_weak_total;
    int                      num_features;
    int                      window_w, window_h;
    const vj_stage_fixed_t  *stages;
    const vj_wc_fixed_t     *weak_classifiers;
    const vj_feature_fixed_t *features_fixed;  /* Step 2: weight codes */
} vj_cascade_fixed_t;

/* Build fixed-point cascade from original float cascade (heap alloc, call once). */
#ifdef __cplusplus
extern "C" {
#endif

vj_cascade_fixed_t *vj_cascade_fixed_build(const vj_cascade_t *src);
void                vj_cascade_fixed_free(vj_cascade_fixed_t *fc);

/* ---------- Step 3: precomputed scaled coordinates (one scale level) ----------
 *
 * For a given window size (win_w × win_h), every rect's position/size is fixed:
 *   dx = (int)(rc->x * scale_x)   offset from window origin
 *   dy = (int)(rc->y * scale_y)
 *   rw = max(1, (int)(rc->w * scale_x))
 *   rh = max(1, (int)(rc->h * scale_y))
 *
 * Build once per scale level (outside the window scan loop).
 * ROM capacity: ~68 KB per scale × ~20 levels ≈ 1.33 MB (see QUANT_ANALYSIS).
 */
typedef struct {
    int16_t dx, dy;   /* scaled x/y offset from window origin */
    int16_t rw, rh;   /* scaled width / height (≥ 1) */
} vj_scaled_rect_t;

typedef struct {
    vj_scaled_rect_t rects[VJ_MAX_RECTS];
    /* num_rects is read from features_fixed — same value, not duplicated */
} vj_scaled_feature_t;

/* Build scaled-coordinate table for one scale level.
 * Returns heap-allocated array[src->num_features]; caller frees with
 * vj_scaled_features_free(). */
vj_scaled_feature_t *vj_scaled_features_build(const vj_cascade_t *src,
                                               int win_w, int win_h);
void                 vj_scaled_features_free(vj_scaled_feature_t *sf);

/*
 * Fixed-point window evaluator (Steps 1+2+3+4) — fully integer, no sqrtf.
 *
 * vs float golden:
 *   - inv_area absorbed (feat_val_raw not divided)
 *   - threshold: Q1.15 quantised
 *   - left/right_val, stage_sum, stage_threshold: Q6.10
 *   - weight multiply replaced by shift/negate via weight_code
 *   - coord scaling: precomputed lookup (no float multiply at runtime)
 *   - sqrtf eliminated: sign-split uint64 square compare (VJ_CMP_SHIFT=2)
 *     va_sq = sqsum*area - sum^2  (int64, = variance*area^2)
 *     |L|^2 vs |T|^2*va_sq in uint64 only, no wide_t/ap_int/__int128
 *
 * scaled_feats must correspond to the current win_w/win_h scale level.
 */
int vj_evaluate_window_fixed(const vj_cascade_fixed_t *fc,
                             const vj_scaled_feature_t *scaled_feats,
                             const uint32_t *ii, const uint64_t *sii,
                             int ii_stride,
                             int win_x, int win_y, int win_w, int win_h);

#ifdef __cplusplus
}
#endif

#endif /* VJ_FIXED_H */
