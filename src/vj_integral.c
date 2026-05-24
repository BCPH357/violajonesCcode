#include "vj_integral.h"
#include <string.h>

void vj_compute_integral_images(const uint8_t *src, int w, int h,
                                uint32_t *ii, uint64_t *sii)
{
    int stride = w + 1;

    /* Zero the first row */
    if (ii)  memset(ii,  0, (size_t)stride * sizeof(uint32_t));
    if (sii) memset(sii, 0, (size_t)stride * sizeof(uint64_t));

    for (int y = 0; y < h; y++) {
        uint32_t row_sum    = 0;
        uint64_t row_sum_sq = 0;

        /* First column of each row is zero (padding) */
        int dst_row = (y + 1) * stride;
        if (ii)  ii[dst_row]  = 0;
        if (sii) sii[dst_row] = 0;

        for (int x = 0; x < w; x++) {
            uint32_t val = src[y * w + x];
            row_sum    += val;
            row_sum_sq += (uint64_t)val * val;

            int idx = dst_row + x + 1;
            int above = idx - stride;

            if (ii)  ii[idx]  = row_sum    + ii[above];
            if (sii) sii[idx] = row_sum_sq + sii[above];
        }
    }
}
