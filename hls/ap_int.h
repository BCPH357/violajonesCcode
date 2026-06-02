/*
 * ap_int.h — local PC stub for Vitis HLS ap_int<N>
 *
 * Backed by __int128 so all operator semantics are correct.
 * Used ONLY for local __SYNTHESIS__ syntax verification.
 * Vitis HLS overrides this with the real Xilinx ap_int.h at synthesis time.
 */
#pragma once
#include <stdint.h>

template<int W>
class ap_int {
    __int128 v;
public:
    ap_int() : v(0) {}
    ap_int(int64_t  x) : v((__int128)x) {}
    ap_int(int32_t  x) : v((__int128)x) {}
    ap_int(__int128 x) : v(x) {}

    ap_int operator*(const ap_int &o) const { return ap_int(v * o.v); }
    bool   operator<(const ap_int &o) const { return v < o.v; }
    bool   operator>(const ap_int &o) const { return v > o.v; }
    bool   operator==(const ap_int &o) const { return v == o.v; }
};
