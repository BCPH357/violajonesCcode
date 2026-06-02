/*
 * minimal_a_baseline.cpp — 純加法 baseline
 *
 * 測試目的：確認 Vitis HLS 環境本身（clang reflow）沒問題。
 * 無任何 wide type、無大 const 表、無複雜控制流。
 * 若此檔也 crash，問題在環境設定，與設計無關。
 */

int minimal_top(int a, int b, int c, int d)
{
#pragma HLS INTERFACE s_axilite port=a      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=b      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=c      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=d      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    return a + b + c + d;
}
