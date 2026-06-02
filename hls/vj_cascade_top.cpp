/*
 * vj_cascade_top.cpp — Vitis HLS AXI-Lite top function (rung1)
 *
 * rung1 change: call vj_evaluate_window_fixed_flat() instead of
 * vj_evaluate_window_fixed().  The three lookup tables are passed as
 * independent const-array parameters (g_stages, g_weak_classifiers,
 * g_features_fixed) rather than via the &g_cascade struct pointer.
 * g_cascade is no longer referenced by the top function or its callees.
 *
 * PC compilation (no __SYNTHESIS__):
 *   g++ -std=c++11 -Isrc -Ihls \
 *       hls/vj_cascade_top.cpp src/vj_fixed.cpp hls/vj_tb.cpp \
 *       -o hls/test_rung1 -static-libstdc++ -static-libgcc
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

    return vj_evaluate_window_fixed_flat(
        g_stages, 25, g_weak_classifiers, g_features_fixed, g_scaled,
        g_test_ii, g_test_sii, TEST_II_STRIDE,
        win_x, win_y, win_w, win_h);
}