/*
 * minimal_i_runtime_bound.cpp — 外層上界改為透過指標讀的 runtime 值
 *
 * 測試目的：H（外層 for (s < 10) 常數，PASS）vs I（外層上界從指標讀，HLS 不知 trip count）。
 * 唯一變數：外層迴圈上界從 compile-time 常數 10 → runtime pointer read fc->num_stages。
 *
 * 模擬完整版 vj_evaluate_window_fixed 的 for (s < fc->num_stages) pattern：
 *   fc 是 function 內的 local pointer，HLS 無法在 elaboration 時確定 num_stages 值，
 *   只能保留成真正的 RTL 控制迴圈（不 unroll）。
 * 同時用 if (s >= 10) break 把 runtime 執行次數控制在 10，
 * 確保規模與 H 相同，隔離「runtime 上界」這個單一變數。
 *
 * 若 H PASS、I crash：問題確認鎖定在「外層上界為 runtime 不可知值」觸發 clang reflow bug。
 * 若 I 也 PASS：差異點在其他地方（wrapper 層數、win_w/win_h 是否為參數等）。
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

    /* 模擬完整版「上界透過指標讀」的 pattern */
    const vj_cascade_fixed_t *fc = &g_cascade;   /* 取指標，HLS 不追蹤 fc->num_stages */
    int num_stages = fc->num_stages;              /* runtime read，HLS 不知道值是 25 */

    for (int s = 0; s < num_stages; s++) {
        if (s >= 10) break;   /* 把 runtime 執行次數限在 10，隔離規模干擾 */

        const vj_stage_fixed_t *stage = &fc->stages[s];
        int32_t stage_sum = 0;

        for (int w = 0; w < stage->num_weak; w++) {
            int idx = stage->weak_start_idx + w;
            const vj_wc_fixed_t       *wc    = &fc->weak_classifiers[idx];
            const vj_feature_fixed_t  *feat  = &fc->features_fixed[wc->feature_idx];
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
