/*
 * ap_int.h — local stub for Vitis HLS ap_int<N>
 *
 * PURPOSE: syntax verification only — lets g++ compile the __SYNTHESIS__
 *          code path in vj_fixed.c without Vitis HLS installed.
 *
 * NOT the real Vitis HLS ap_int<N>:
 *   - Real ap_int<N> synthesises an N-bit hardware register.
 *   - This stub backs every instantiation with __int128 (sufficient for
 *     N <= 82 as used in cmp_lhs_lt_rhs).
 *   - No bit-width truncation, no hardware behaviour.
 *
 * When Vitis HLS compiles vj_fixed.c it uses its own <ap_int.h> from the
 * tool's include path, which overrides this stub.  Locally, -Ihls picks
 * up this file.
 *
 * Operations implemented (exactly what cmp_lhs_lt_rhs uses):
 *   construction from any integer type
 *   operator*   (multiplication)
 *   operator<   (less-than)
 *   operator>   (greater-than)
 */
#pragma once
#include <cstdint>

template<int W>
class ap_int {
    __int128 v_;
    explicit ap_int(__int128 raw) : v_(raw) {}
public:
    ap_int() : v_(0) {}

    template<typename T>
    ap_int(T x) : v_(static_cast<__int128>(x)) {}

    ap_int operator*(const ap_int& o) const { return ap_int(v_ * o.v_); }
    bool   operator<(const ap_int& o) const { return v_ < o.v_; }
    bool   operator>(const ap_int& o) const { return v_ > o.v_; }
};
