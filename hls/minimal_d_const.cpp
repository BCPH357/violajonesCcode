/*
 * minimal_d_const.cpp — 觸碰 hls_test_data.h 的大 const 表
 *
 * 測試目的：確認 2913-entry 的 cascade 表 + 3969-entry 的 II 陣列
 * 是否讓 clang reflow crash（巨大 static const 展開問題）。
 * 邏輯刻意極簡，避免干擾診斷。
 * 若 A/B/C 都過、D crash：問題在大 const 表的 LLVM IR 展開。
 *
 * Vitis HLS 需額外設定 Include Path：加入 <project>/src/
 * （hls_test_data.h 會 #include "vj_fixed.h"，vj_fixed.h 在 src/）
 */
#include "hls_test_data.h"

int minimal_top(int a, int b, int c, int d)
{
#pragma HLS INTERFACE s_axilite port=a      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=b      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=c      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=d      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    return a
         + (int)g_stages[b % 25].num_weak
         + (int)g_test_ii[c & 0xFFF]
         + d;
}
