/*
 * minimal_g_5stages.cpp — 前 5 個 stage（136 weak classifiers）
 *
 * 測試目的：從 F（2 stages, 25 wc, PASS）向上擴展，
 * 確認 5 stages / 136 wc 是否讓 clang reflow crash。
 *
 * 結構與 F 完全同形（外層 stage 迴圈 + early-rejection + runtime 內層邊界）。
 * 唯一變數：外層迴圈上界 5（F 是 2），涵蓋更多 stage->num_weak 變化值。
 *
 * Stage weak classifier 數（累計 136）：
 *   stage 0:  9 wc  (wc idx   0-  8)
 *   stage 1: 16 wc  (wc idx   9- 24)
 *   stage 2: 27 wc  (wc idx  25- 51)
 *   stage 3: 32 wc  (wc idx  52- 83)
 *   stage 4: 52 wc  (wc idx  84-135)
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

    for (int s = 0; s < 5; s++) {
        const vj_stage_fixed_t *stage = &g_cascade.stages[s];
        int32_t stage_sum = 0;

        for (int w = 0; w < stage->num_weak; w++) {
            int idx = stage->weak_start_idx + w;
            const vj_wc_fixed_t       *wc    = &g_cascade.weak_classifiers[idx];
            const vj_feature_fixed_t  *feat  = &g_cascade.features_fixed[wc->feature_idx];
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
            return 0;
    }
    return 1;
}
