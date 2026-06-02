/*
 * minimal_c_cmp.cpp — 完整複製 cmp_lhs_lt_rhs 四分支邏輯
 *
 * 測試目的：確認「四分支符號安全平方比較 + ap_int<74>」結構是否讓 clang reflow crash。
 * 邏輯與 src/vj_fixed.cpp 的 cmp_lhs_lt_rhs() 完全等價，但入口改用 AXI-Lite port。
 * 若 B 過、C crash：問題在四分支 + wide_t 組合，而非 ap_int 本身。
 */
#include "ap_int.h"

typedef ap_int<74> wide_t;

int minimal_top(int L_in, int T_in, int V_in, int A_in)
{
#pragma HLS INTERFACE s_axilite port=L_in   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=T_in   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=V_in   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=A_in   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    long L = L_in, V = V_in, A = A_in;
    int  T = T_in;
    int  result;

    if (V <= 0) {
        result = (L < (long)T * A) ? 1 : 0;
    } else if (T > 0) {
        wide_t Lw = L, Tw = T, Vw = V;
        result = ((L < 0) || (Lw * Lw < Tw * Tw * Vw)) ? 1 : 0;
    } else if (T < 0) {
        wide_t Lw = L, Tw = T, Vw = V;
        result = ((L < 0) && (Lw * Lw > Tw * Tw * Vw)) ? 1 : 0;
    } else {
        result = (L < 0) ? 1 : 0;
    }
    return result;
}
