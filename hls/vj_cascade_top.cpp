/*
 * vj_cascade_top.cpp — Vitis HLS AXI-Lite top function (A version)
 *
 * HLS entry point: vj_cascade_top()
 *   - All ports mapped to AXI-Lite bundle CTRL
 *   - II/sII arrays and cascade tables baked in from hls_test_data.h
 *   - Delegates to vj_evaluate_window_fixed() in vj_fixed.c
 *
 * PC compilation (no __SYNTHESIS__):
 *   g++ -std=c++11 -Isrc -Ihls -o hls/test_top \
 *       hls/vj_cascade_top.cpp src/vj_fixed.c
 *   ./hls/test_top   -> both PASS expected
 */

#ifdef __SYNTHESIS__
#  include "ap_int.h"
#endif
#include "vj_fixed.h"
#include "hls_test_data.h"

int vj_cascade_top(int win_x, int win_y, int win_w, int win_h)
{
#pragma HLS INTERFACE s_axilite port=win_x   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=win_y   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=win_w   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=win_h   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return  bundle=CTRL

    return vj_evaluate_window_fixed(&g_cascade, g_scaled,
                                    g_test_ii, g_test_sii, TEST_II_STRIDE,
                                    win_x, win_y, win_w, win_h);
}

#ifndef __SYNTHESIS__
#include <stdio.h>
int main(void)
{
    int r1 = vj_cascade_top(HLS_PASS_X,   HLS_PASS_Y,   HLS_PASS_W,   HLS_PASS_H);
    int r2 = vj_cascade_top(HLS_REJECT_X, HLS_REJECT_Y, HLS_REJECT_W, HLS_REJECT_H);
    printf("pass   (%d,%d,%d,%d) -> %d  expected %d  %s\n",
           HLS_PASS_X, HLS_PASS_Y, HLS_PASS_W, HLS_PASS_H,
           r1, HLS_PASS_EXPECTED, r1 == HLS_PASS_EXPECTED ? "PASS" : "FAIL");
    printf("reject (%d,%d,%d,%d) -> %d  expected %d  %s\n",
           HLS_REJECT_X, HLS_REJECT_Y, HLS_REJECT_W, HLS_REJECT_H,
           r2, HLS_REJECT_EXPECTED, r2 == HLS_REJECT_EXPECTED ? "PASS" : "FAIL");
    return (r1 == HLS_PASS_EXPECTED && r2 == HLS_REJECT_EXPECTED) ? 0 : 1;
}
#endif
