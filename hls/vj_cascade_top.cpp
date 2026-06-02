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

#include "vj_fixed.h"
#include "hls_test_data.h"

int vj_cascade_top(int win_x, int win_y, int win_w, int win_h)
{
#pragma HLS INTERFACE s_axilite port=win_x   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=win_y   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=win_w   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=win_h   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return  bundle=CTRL

    /* Disable automatic array partition to prevent clang OOM crash during
     * HLS reflow (Xilinx known bug, FINN issue #1214). */
#pragma HLS array_partition variable=g_test_ii          off=true
#pragma HLS array_partition variable=g_test_sii         off=true
#pragma HLS array_partition variable=g_scaled_feats     off=true
#pragma HLS array_partition variable=g_features_fixed   off=true
#pragma HLS array_partition variable=g_weak_classifiers off=true
#pragma HLS array_partition variable=g_stages           off=true

    return vj_evaluate_window_fixed(&g_cascade, g_scaled,
                                    g_test_ii, g_test_sii, TEST_II_STRIDE,
                                    win_x, win_y, win_w, win_h);
}