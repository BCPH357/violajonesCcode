/*
 * minimal_b_apint.cpp — ap_int<74> 三運算元乘法
 *
 * 測試目的：確認 ap_int<74> 本身（大位寬乘法 + 截取）是否讓 clang reflow crash。
 * 若 A 過、B crash：問題在 ap_int<74> 乘法電路生成。
 */
#include "ap_int.h"

int minimal_top(int a, int b, int c, int d)
{
#pragma HLS INTERFACE s_axilite port=a      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=b      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=c      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=d      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    ap_int<74> x = a, y = b, z = c;
    ap_int<74> r = x * y * z;
    return (int)(r & 0xFFFF) + d;
}
