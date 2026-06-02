/*
 * minimal_f_twostages.cpp — stage 0 + stage 1 (9 + 16 = 25 weak classifiers)
 *
 * 測試目的：加上外層 stage 迴圈 + early-rejection，
 * 確認「兩層迴圈 + early return + const 表 + wide_t」組合是否讓 clang reflow crash。
 *
 * 若 E 過、F crash：問題在外層 stage 迴圈本身或 early-rejection 的控制流。
 * 若 F 也過：crash 的根因是 25 stages 的完整規模（total 2913 wc），
 *            下一步縮減方向：逐步加 stage 數（G=5 stages、H=10 stages …）。
 *
 * Vitis HLS include path 需要加：<repo>/src/
 */
#include "hls_test_data.h"   /* g_cascade, g_scaled, g_test_ii, g_test_sii, TEST_II_STRIDE */
#include "vj_integral.h"     /* vj_rect_sum(), vj_rect_sum_sq() */

#ifdef __SYNTHESIS__
#  include "ap_int.h"
   typedef ap_int<74> wide_t;
#else
   typedef __int128 wide_t;
#endif

/* 直接複製自 src/vj_fixed.cpp */
static inline int cmp_lhs_lt_rhs(int64_t L, int32_t T,
                                  int64_t va_sq, int64_t area)
{
    if (va_sq <= 0)
        return L < (int64_t)T * area;
    if (T > 0) {
        wide_t Lw = L, Tw = T, Vw = va_sq;
        return (L < 0) || (Lw * Lw < Tw * Tw * Vw);
    }
    if (T < 0) {
        wide_t Lw = L, Tw = T, Vw = va_sq;
        return (L < 0) && (Lw * Lw > Tw * Tw * Vw);
    }
    return L < 0;
}

int minimal_top(int win_x, int win_y)
{
#pragma HLS INTERFACE s_axilite port=win_x  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=win_y  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    const int win_w = 50;
    const int win_h = 50;
    const int stride = TEST_II_STRIDE;

    uint32_t sum   = vj_rect_sum   (g_test_ii,  stride, win_x, win_y, win_w, win_h);
    uint64_t sqsum = vj_rect_sum_sq(g_test_sii, stride, win_x, win_y, win_w, win_h);
    int64_t  area  = (int64_t)(win_w * win_h);
    int64_t  va_sq = (int64_t)sqsum * area - (int64_t)sum * (int64_t)sum;

    /* 跑 stage 0 + stage 1，含 early-rejection */
    for (int s = 0; s < 2; s++) {
        const vj_stage_fixed_t *stage = &g_cascade.stages[s];
        int32_t stage_sum = 0;

        for (int w = 0; w < stage->num_weak; w++) {
            int idx = stage->weak_start_idx + w;
            const vj_wc_fixed_t      *wc    = &g_cascade.weak_classifiers[idx];
            const vj_feature_fixed_t *feat  = &g_cascade.features_fixed[wc->feature_idx];
            const vj_scaled_feature_t *sfeat = &g_scaled[wc->feature_idx];

            int32_t feat_val_raw = 0;
            for (int r = 0; r < feat->num_rects; r++) {
                int rx = win_x + sfeat->rects[r].dx;
                int ry = win_y + sfeat->rects[r].dy;
                int rw = sfeat->rects[r].rw;
                int rh = sfeat->rects[r].rh;
                int32_t rs = (int32_t)vj_rect_sum(g_test_ii, stride, rx, ry, rw, rh);
                feat_val_raw += vj_apply_weight(rs, feat->rects[r].weight_code);
            }

            int64_t L    = (int64_t)feat_val_raw << VJ_Q_THRESH;
            int     pass = cmp_lhs_lt_rhs(L, wc->threshold_q15, va_sq, area);
            stage_sum   += pass ? wc->left_val_q10 : wc->right_val_q10;
        }

        if (stage_sum < stage->stage_threshold_q10)
            return 0;   /* early rejection */
    }
    return 1;
}
