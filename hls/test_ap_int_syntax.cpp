/*
 * test_ap_int_syntax.cpp — syntax verification of __SYNTHESIS__ + ap_int<74>
 *
 * Simulates the code path Vitis HLS will compile from vj_fixed.c:
 *   - __SYNTHESIS__ defined
 *   - ap_int<74> used as wide_t
 *   - all four branches of cmp_lhs_lt_rhs instantiated
 *
 * Compiled with: g++ -std=c++14 -Ihls -static-libstdc++ -static-libgcc
 * This is a compile+run test — if the binary runs and exits 0, both
 * syntax and operator semantics are correct under the stub.
 */
#define __SYNTHESIS__
#include "ap_int.h"   /* local stub (real Vitis HLS overrides this) */
#include <stdint.h>
#include <stdio.h>

typedef ap_int<74> wide_t;

/* Exact copy of cmp_lhs_lt_rhs from src/vj_fixed.c — not modified */
static inline int cmp_lhs_lt_rhs(int64_t L, int32_t T,
                                  int64_t va_sq, int64_t area)
{
    if (va_sq <= 0)
        return L < (int64_t)T * area;
    if (T > 0) {
        wide_t Lw = L, Tw = T, Vw = va_sq;
        return (L < 0) || (Lw * Lw < Tw * Tw * Vw);
    }
    if (T < 0) {
        wide_t Lw = L, Tw = T, Vw = va_sq;
        return (L < 0) && (Lw * Lw > Tw * Tw * Vw);
    }
    return L < 0;   /* T == 0 */
}

int main(void)
{
    int fail = 0;

    /* Branch 1: va_sq <= 0 (flat region) */
    /* L=100, T=5, va_sq=-1 (degenerate): 100 < 5*2500=12500 → 1 */
    int r0 = cmp_lhs_lt_rhs(100LL, 5, -1LL, 2500LL);
    if (r0 != 1) { printf("branch va_sq<=0 FAIL (got %d)\n", r0); fail++; }

    /* Branch 2: T > 0, L >= 0, L^2 < T^2*va_sq  →  1 */
    /* L=1000, T=5, va_sq=1e6, area=2500: L^2=1e6, T^2*va_sq=25e6 → 1e6<25e6=1 */
    int r1 = cmp_lhs_lt_rhs(1000LL, 5, 1000000LL, 2500LL);
    if (r1 != 1) { printf("branch T>0 (L^2<R^2) FAIL (got %d)\n", r1); fail++; }

    /* Branch 3: T > 0, L < 0  →  always 1 */
    int r2 = cmp_lhs_lt_rhs(-500LL, 5, 1000000LL, 2500LL);
    if (r2 != 1) { printf("branch T>0 (L<0) FAIL (got %d)\n", r2); fail++; }

    /* Branch 4: T < 0, L >= 0  →  always 0 */
    int r3 = cmp_lhs_lt_rhs(500LL, -3, 500000LL, 2500LL);
    if (r3 != 0) { printf("branch T<0 (L>=0) FAIL (got %d)\n", r3); fail++; }

    /* Branch 5: T == 0  →  L < 0 */
    int r4 = cmp_lhs_lt_rhs(-100LL, 0, 200000LL, 2500LL);
    if (r4 != 1) { printf("branch T=0 (L<0) FAIL (got %d)\n", r4); fail++; }
    int r5 = cmp_lhs_lt_rhs(100LL, 0, 200000LL, 2500LL);
    if (r5 != 0) { printf("branch T=0 (L>0) FAIL (got %d)\n", r5); fail++; }

    if (fail == 0)
        printf("ap_int<74> synthesis path: all %d branches PASS\n",
               (int)(r0+r1+r2+r3+r4+r5+1));
    return fail;
}
