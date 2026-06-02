#include "vj_fixed.h"
#include "vj_integral.h"

/* ------------------------------------------------------------------ */
/* Build helpers: malloc/free — excluded from HLS synthesis call graph  */
#ifndef __SYNTHESIS__
#include <stdlib.h>

static inline int32_t to_q(float v, int q_bits)
{
    float scaled = v * (float)(1 << q_bits);
    return (int32_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
}

static inline int8_t weight_to_code(float w)
{
    if (w < 0.0f) return VJ_WCODE_NEG1;
    if (w < 2.5f) return VJ_WCODE_POS2;
    return VJ_WCODE_POS3;
}

/* ------------------------------------------------------------------ */
vj_cascade_fixed_t *vj_cascade_fixed_build(const vj_cascade_t *src)
{
    vj_cascade_fixed_t *fc = (vj_cascade_fixed_t *)malloc(sizeof(*fc));
    if (!fc) return NULL;

    vj_stage_fixed_t *stages =
        (vj_stage_fixed_t *)malloc((size_t)src->num_stages * sizeof(vj_stage_fixed_t));
    vj_wc_fixed_t *wcs =
        (vj_wc_fixed_t *)malloc((size_t)src->num_weak_total * sizeof(vj_wc_fixed_t));
    vj_feature_fixed_t *feats =
        (vj_feature_fixed_t *)malloc((size_t)src->num_features * sizeof(vj_feature_fixed_t));

    if (!stages || !wcs || !feats) {
        free(stages); free(wcs); free(feats); free(fc);
        return NULL;
    }

    for (int s = 0; s < src->num_stages; s++) {
        stages[s].num_weak           = src->stages[s].num_weak;
        stages[s].weak_start_idx     = src->stages[s].weak_start_idx;
        stages[s].stage_threshold_q10 =
            to_q(src->stages[s].stage_threshold, VJ_Q_LRVAL);
    }

    for (int i = 0; i < src->num_weak_total; i++) {
        wcs[i].feature_idx   = src->weak_classifiers[i].feature_idx;
        wcs[i].threshold_q15 = to_q(src->weak_classifiers[i].threshold,  VJ_Q_THRESH);
        wcs[i].left_val_q10  = to_q(src->weak_classifiers[i].left_val,   VJ_Q_LRVAL);
        wcs[i].right_val_q10 = to_q(src->weak_classifiers[i].right_val,  VJ_Q_LRVAL);
    }

    for (int f = 0; f < src->num_features; f++) {
        const vj_feature_t *sf = &src->features[f];
        feats[f].num_rects = sf->num_rects;
        for (int r = 0; r < sf->num_rects; r++) {
            feats[f].rects[r].x           = sf->rects[r].x;
            feats[f].rects[r].y           = sf->rects[r].y;
            feats[f].rects[r].w           = sf->rects[r].w;
            feats[f].rects[r].h           = sf->rects[r].h;
            feats[f].rects[r].weight_code = weight_to_code(sf->rects[r].weight);
        }
    }

    fc->num_stages       = src->num_stages;
    fc->num_weak_total   = src->num_weak_total;
    fc->num_features     = src->num_features;
    fc->window_w         = src->window_w;
    fc->window_h         = src->window_h;
    fc->stages           = stages;
    fc->weak_classifiers = wcs;
    fc->features_fixed   = feats;
    return fc;
}

void vj_cascade_fixed_free(vj_cascade_fixed_t *fc)
{
    if (!fc) return;
    free((void *)fc->stages);
    free((void *)fc->weak_classifiers);
    free((void *)fc->features_fixed);
    free(fc);
}

/* ------------------------------------------------------------------ */
/* Step 3: build precomputed scaled-coordinate table for one scale level */

vj_scaled_feature_t *vj_scaled_features_build(const vj_cascade_t *src,
                                               int win_w, int win_h)
{
    vj_scaled_feature_t *sf =
        (vj_scaled_feature_t *)malloc((size_t)src->num_features * sizeof(vj_scaled_feature_t));
    if (!sf) return NULL;

    float scale_x = (float)win_w / (float)src->window_w;
    float scale_y = (float)win_h / (float)src->window_h;

    for (int f = 0; f < src->num_features; f++) {
        const vj_feature_t *feat = &src->features[f];
        for (int r = 0; r < feat->num_rects; r++) {
            const vj_rect_t *rc = &feat->rects[r];
            int dx = (int)(rc->x * scale_x);
            int dy = (int)(rc->y * scale_y);
            int rw = (int)(rc->w * scale_x); if (rw < 1) rw = 1;
            int rh = (int)(rc->h * scale_y); if (rh < 1) rh = 1;
            sf[f].rects[r].dx = (int16_t)dx;
            sf[f].rects[r].dy = (int16_t)dy;
            sf[f].rects[r].rw = (int16_t)rw;
            sf[f].rects[r].rh = (int16_t)rh;
        }
    }
    return sf;
}

void vj_scaled_features_free(vj_scaled_feature_t *sf)
{
    free(sf);
}
#endif /* !__SYNTHESIS__ */

/* ------------------------------------------------------------------ */
/*
 * Step 4: sign-safe square comparison replacing L < T * sqrt(va_sq).
 *
 * Rewritten to eliminate wide_t/ap_int/__int128: pure int64_t/uint64_t.
 *
 * Sign handling at outer level (L_sign, R_sign = T_sign since stdev >= 0).
 * Same-sign path: compare |L|^2 vs |T|^2 * va_sq via right-shift VJ_CMP_SHIFT.
 *
 * VJ_CMP_SHIFT=2 (shift each |L| factor by 1 bit, va_sq by 2 bits):
 *   lhs = (|L|>>1)^2              empirical max ~59-bit, safe in uint64
 *   rhs = |T|^2 * (va_sq>>2)     empirical max ~61-bit, safe in uint64;
 *                                 provably safe up to theoretical worst-case
 *                                 va_sq ~1e11 (|T|_max^2 * 1e11/4 < 2^64)
 *
 * Empirical bounds (testt.jpg, 169 windows x 2913 classifiers):
 *   max |L|=1,602,617,344 (31-bit)  max |T|=21,647 (15-bit)
 *   max va_sq=11,797,284,931 (34-bit)
 * Consistency vs old __int128 version: 100.000% on all 169 windows.
 */
#define VJ_CMP_SHIFT 2

static inline int cmp_lhs_lt_rhs(int64_t L, int32_t T,
                                  int64_t va_sq, int64_t area)
{
    if (va_sq <= 0)
        return L < (int64_t)T * area;

    int L_sign = (L > 0) - (L < 0);
    int R_sign = (T > 0) - (T < 0);

    if (L_sign < R_sign) return 1;
    if (L_sign > R_sign) return 0;
    if (L_sign == 0)     return 0;

    uint64_t aL  = (uint64_t)(L > 0 ? L : -L);
    uint64_t aT  = (uint64_t)(T > 0 ? (int64_t)T : -(int64_t)T);
    uint64_t lhs = (aL >> (VJ_CMP_SHIFT/2)) * (aL >> (VJ_CMP_SHIFT/2));
    uint64_t rhs = aT * aT * ((uint64_t)va_sq >> VJ_CMP_SHIFT);

    return (L_sign > 0) ? (lhs < rhs) : (lhs > rhs);
}

/* ------------------------------------------------------------------ */
/* Fixed-point window evaluator (Steps 1+2+3+4) ?? fully integer, no sqrtf */

int vj_evaluate_window_fixed(const vj_cascade_fixed_t *fc,
                             const vj_scaled_feature_t *scaled_feats,
                             const uint32_t *ii, const uint64_t *sii,
                             int ii_stride,
                             int win_x, int win_y, int win_w, int win_h)
{
    uint32_t sum   = vj_rect_sum   (ii,  ii_stride, win_x, win_y, win_w, win_h);
    uint64_t sqsum = vj_rect_sum_sq(sii, ii_stride, win_x, win_y, win_w, win_h);

    int64_t area  = (int64_t)(win_w * win_h);
    /* va_sq = variance * area^2: always >= 0, fits int64 */
    int64_t va_sq = (int64_t)sqsum * area
                  - (int64_t)sum   * (int64_t)sum;

    for (int s = 0; s < fc->num_stages; s++) {
#pragma HLS UNROLL off
        const vj_stage_fixed_t *stage = &fc->stages[s];
        int32_t stage_sum = 0;

        for (int w = 0; w < stage->num_weak; w++) {
#pragma HLS UNROLL off
            const vj_wc_fixed_t       *wc    =
                &fc->weak_classifiers[stage->weak_start_idx + w];
            const vj_feature_fixed_t  *feat  =
                &fc->features_fixed[wc->feature_idx];
            const vj_scaled_feature_t *sfeat =
                &scaled_feats[wc->feature_idx];

            int32_t feat_val_raw = 0;
            for (int r = 0; r < feat->num_rects; r++) {
                int rx = win_x + sfeat->rects[r].dx;
                int ry = win_y + sfeat->rects[r].dy;
                int rw = sfeat->rects[r].rw;
                int rh = sfeat->rects[r].rh;
                int32_t rs = (int32_t)vj_rect_sum(ii, ii_stride, rx, ry, rw, rh);
                feat_val_raw += vj_apply_weight(rs, feat->rects[r].weight_code);
            }

            /* Step 4: L < T*sqrt(va_sq) without sqrtf */
            int64_t L = (int64_t)feat_val_raw << VJ_Q_THRESH;
            int lhs_lt_rhs = cmp_lhs_lt_rhs(L, wc->threshold_q15, va_sq, area);

            stage_sum += lhs_lt_rhs ? wc->left_val_q10 : wc->right_val_q10;
        }

        if (stage_sum < stage->stage_threshold_q10)
            return 0;
    }
    return 1;
}
