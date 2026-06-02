/*
 * verify_cmp.cpp — 驗證 int64-only cmp_lhs_lt_rhs 相對 __int128 版的一致率
 *
 * Compile (MSYS2):
 *   g++ -std=c++14 -Isrc -Ihls \
 *       -o hls/verify_cmp.exe hls/verify_cmp.cpp \
 *       -static-libstdc++ -static-libgcc
 */
#include "hls_test_data.h"
#include "vj_integral.h"
#include "vj_fixed.h"
#include <stdio.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Old: uses __int128 (golden reference) */
static int cmp_old(int64_t L, int32_t T, int64_t va_sq, int64_t area)
{
    typedef __int128 wide_t;
    if (va_sq <= 0) return L < (int64_t)T * area;
    if (T > 0) { wide_t a=L,b=T,v=va_sq; return (L<0)||(a*a<b*b*v); }
    if (T < 0) { wide_t a=L,b=T,v=va_sq; return (L<0)&&(a*a>b*b*v); }
    return L < 0;
}

/* ------------------------------------------------------------------ */
/* New: int64-only, runtime K (for sweep; production uses compile-time K) */
static int cmp_new_k(int64_t L, int32_t T, int64_t va_sq, int64_t area, int K)
{
    if (va_sq <= 0) return L < (int64_t)T * area;
    int Ls = (L > 0) - (L < 0);
    int Rs = (T > 0) - (T < 0);
    if (Ls < Rs) return 1;
    if (Ls > Rs) return 0;
    if (Ls == 0) return 0;   /* both zero: 0 < 0 is false */
    uint64_t aL = (uint64_t)(L > 0 ? L : -L);
    uint64_t aT = (uint64_t)(T > 0 ? (int64_t)T : -(int64_t)T);
    uint64_t lhs = (aL >> K) * (aL >> K);
    uint64_t rhs = aT * aT * ((uint64_t)va_sq >> (2 * K));
    return (Ls > 0) ? (lhs < rhs) : (lhs > rhs);
}

/* ------------------------------------------------------------------ */
static int eval_old(int wx, int wy)
{
    const int WW=50, WH=50, ST=TEST_II_STRIDE;
    uint32_t s  = vj_rect_sum   (g_test_ii,  ST, wx, wy, WW, WH);
    uint64_t sq = vj_rect_sum_sq(g_test_sii, ST, wx, wy, WW, WH);
    int64_t area = (int64_t)(WW * WH);
    int64_t va   = (int64_t)sq * area - (int64_t)s * (int64_t)s;
    for (int i = 0; i < g_cascade.num_stages; i++) {
        const vj_stage_fixed_t *st = &g_cascade.stages[i];
        int32_t ss = 0;
        for (int j = 0; j < st->num_weak; j++) {
            const vj_wc_fixed_t       *wc = &g_cascade.weak_classifiers[st->weak_start_idx+j];
            const vj_feature_fixed_t  *f  = &g_cascade.features_fixed[wc->feature_idx];
            const vj_scaled_feature_t *sf = &g_scaled[wc->feature_idx];
            int32_t fv = 0;
            for (int r = 0; r < f->num_rects; r++) {
                int32_t rs = (int32_t)vj_rect_sum(g_test_ii, ST,
                    wx+sf->rects[r].dx, wy+sf->rects[r].dy,
                    sf->rects[r].rw, sf->rects[r].rh);
                fv += vj_apply_weight(rs, f->rects[r].weight_code);
            }
            int64_t L = (int64_t)fv << VJ_Q_THRESH;
            ss += cmp_old(L, wc->threshold_q15, va, area)
                  ? wc->left_val_q10 : wc->right_val_q10;
        }
        if (ss < st->stage_threshold_q10) return 0;
    }
    return 1;
}

static int eval_new(int wx, int wy, int K)
{
    const int WW=50, WH=50, ST=TEST_II_STRIDE;
    uint32_t s  = vj_rect_sum   (g_test_ii,  ST, wx, wy, WW, WH);
    uint64_t sq = vj_rect_sum_sq(g_test_sii, ST, wx, wy, WW, WH);
    int64_t area = (int64_t)(WW * WH);
    int64_t va   = (int64_t)sq * area - (int64_t)s * (int64_t)s;
    for (int i = 0; i < g_cascade.num_stages; i++) {
        const vj_stage_fixed_t *st = &g_cascade.stages[i];
        int32_t ss = 0;
        for (int j = 0; j < st->num_weak; j++) {
            const vj_wc_fixed_t       *wc = &g_cascade.weak_classifiers[st->weak_start_idx+j];
            const vj_feature_fixed_t  *f  = &g_cascade.features_fixed[wc->feature_idx];
            const vj_scaled_feature_t *sf = &g_scaled[wc->feature_idx];
            int32_t fv = 0;
            for (int r = 0; r < f->num_rects; r++) {
                int32_t rs = (int32_t)vj_rect_sum(g_test_ii, ST,
                    wx+sf->rects[r].dx, wy+sf->rects[r].dy,
                    sf->rects[r].rw, sf->rects[r].rh);
                fv += vj_apply_weight(rs, f->rects[r].weight_code);
            }
            int64_t L = (int64_t)fv << VJ_Q_THRESH;
            ss += cmp_new_k(L, wc->threshold_q15, va, area, K)
                  ? wc->left_val_q10 : wc->right_val_q10;
        }
        if (ss < st->stage_threshold_q10) return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
int main(void)
{
    const int IW=62, IH=62, WW=50, WH=50;
    int64_t maxL=0, maxT=0, maxV=0;

    /* Pass 1: find max values across ALL classifiers (no early exit) */
    for (int wy=0; wy<=IH-WH; wy++) for (int wx=0; wx<=IW-WW; wx++) {
        const int ST=TEST_II_STRIDE;
        uint32_t sm = vj_rect_sum   (g_test_ii,  ST, wx, wy, WW, WH);
        uint64_t sq = vj_rect_sum_sq(g_test_sii, ST, wx, wy, WW, WH);
        int64_t area = (int64_t)(WW*WH);
        int64_t va   = (int64_t)sq*area - (int64_t)sm*(int64_t)sm;
        if (va > maxV) maxV = va;
        for (int i=0; i<g_cascade.num_stages; i++) {
            const vj_stage_fixed_t *st = &g_cascade.stages[i];
            for (int j=0; j<st->num_weak; j++) {
                const vj_wc_fixed_t       *wc = &g_cascade.weak_classifiers[st->weak_start_idx+j];
                const vj_feature_fixed_t  *f  = &g_cascade.features_fixed[wc->feature_idx];
                const vj_scaled_feature_t *sf = &g_scaled[wc->feature_idx];
                int32_t fv=0;
                for (int r=0; r<f->num_rects; r++) {
                    int32_t rs = (int32_t)vj_rect_sum(g_test_ii,ST,
                        wx+sf->rects[r].dx,wy+sf->rects[r].dy,
                        sf->rects[r].rw,sf->rects[r].rh);
                    fv += vj_apply_weight(rs, f->rects[r].weight_code);
                }
                int64_t L  = (int64_t)fv << VJ_Q_THRESH;
                int64_t aL = L > 0 ? L : -L;
                int64_t aT = wc->threshold_q15>0
                             ? wc->threshold_q15 : -(int64_t)wc->threshold_q15;
                if (aL > maxL) maxL = aL;
                if (aT > maxT) maxT = aT;
            }
        }
    }

    /* Bit-width analysis */
    int Lb=0; { int64_t v=maxL; while(v){v>>=1;Lb++;} }
    int Tb=0; { int64_t v=maxT; while(v){v>>=1;Tb++;} }
    int Vb=0; { int64_t v=maxV; while(v){v>>=1;Vb++;} }

    printf("=== Bit-width analysis ===\n");
    printf("Max |L|   = %lld  (%d-bit)\n", (long long)maxL, Lb);
    printf("Max |T|   = %lld  (%d-bit)\n", (long long)maxT, Tb);
    printf("Max va_sq = %lld  (%d-bit)\n", (long long)maxV, Vb);
    printf("|L|^2     bits = %d\n", 2*Lb);
    printf("|T|^2*V   bits = %d\n", 2*Tb+Vb);

    /* Minimum K to avoid lhs overflow: (maxL >> K)^2 <= UINT64_MAX
     * → maxL >> K <= 2^32 → K >= Lb - 32 */
    int k_min_lhs = Lb - 32; if (k_min_lhs < 0) k_min_lhs = 0;
    /* Minimum K to avoid rhs overflow: aT^2 * (va_sq >> 2K) <= UINT64_MAX
     * conservative: use 2*Tb + Vb - 64 bits of shift needed */
    int k_min_rhs = (2*Tb + Vb - 64 + 1) / 2; if (k_min_rhs < 0) k_min_rhs = 0;
    int k_min = k_min_lhs > k_min_rhs ? k_min_lhs : k_min_rhs;
    printf("Min K for no overflow: K_lhs>=%d, K_rhs>=%d → K_min=%d\n",
           k_min_lhs, k_min_rhs, k_min);
    printf("\n");

    /* Pass 2: sweep K from k_min to k_min+4, report consistency */
    int best_K = -1;
    int total = (IW-WW+1)*(IH-WH+1);
    printf("=== Window consistency sweep (total windows = %d) ===\n", total);
    printf("  K  |  N=2K | mismatch |  rate%%\n");
    printf("-----|-------|----------|--------\n");

    for (int K = k_min; K <= k_min+4; K++) {
        int miss = 0;
        for (int wy=0; wy<=IH-WH; wy++) for (int wx=0; wx<=IW-WW; wx++) {
            int ro = eval_old(wx, wy);
            int rn = eval_new(wx, wy, K);
            if (ro != rn) miss++;
        }
        double rate = 100.0 * (total - miss) / total;
        printf("  %2d |   %2d  |   %4d   | %7.3f%%  %s\n",
               K, 2*K, miss, rate, (best_K < 0 && rate >= 99.5) ? "<-- FIRST PASS" : "");
        if (best_K < 0 && rate >= 99.5) best_K = K;
    }
    printf("\n");

    if (best_K < 0) {
        printf("FAIL: no K in [%d, %d] achieves >= 99.5%% consistency.\n",
               k_min, k_min+4);
        printf("      Consult user before proceeding.\n");
        return 1;
    }

    /* Golden vector check with best K */
    int r1o = eval_old(6,6), r2o = eval_old(0,0);
    int r1n = eval_new(6,6,best_K), r2n = eval_new(0,0,best_K);
    printf("=== Golden vectors (K=%d) ===\n", best_K);
    printf("(6,6,50,50) old=%d new=%d  %s\n", r1o, r1n, r1o==r1n?"OK":"FAIL");
    printf("(0,0,50,50) old=%d new=%d  %s\n", r2o, r2n, r2o==r2n?"OK":"FAIL");

    if (r1o != r1n || r2o != r2n) {
        printf("FAIL: golden vector mismatch.\n");
        return 1;
    }

    printf("\nRECOMMEND: #define VJ_CMP_SHIFT_HALF %d  (VJ_CMP_SHIFT=%d)\n",
           best_K, 2*best_K);
    return 0;
}
