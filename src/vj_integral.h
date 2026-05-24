#ifndef VJ_INTEGRAL_H
#define VJ_INTEGRAL_H

#include <stdint.h>

/*
 * Compute integral image and squared integral image.
 *
 * Both output arrays use a (w+1)x(h+1) padded layout with a leading row
 * and column of zeros.  This eliminates boundary checks when computing
 * rectangle sums.
 *
 * ii  must point to (w+1)*(h+1) uint32_t elements.
 * sii must point to (w+1)*(h+1) uint64_t elements.
 *
 * Either ii or sii may be NULL if not needed.
 */
void vj_compute_integral_images(const uint8_t *src, int w, int h,
                                uint32_t *ii, uint64_t *sii);

/*
 * Rectangle sum from a padded integral image.
 * (x,y) is top-left corner, (rw,rh) is width and height.
 * stride = image_width + 1.
 */
static inline uint32_t vj_rect_sum(const uint32_t *ii, int stride,
                                   int x, int y, int rw, int rh)
{
    return ii[(y + rh) * stride + (x + rw)]
         - ii[y        * stride + (x + rw)]
         - ii[(y + rh) * stride +  x      ]
         + ii[y        * stride +  x      ];
}

static inline uint64_t vj_rect_sum_sq(const uint64_t *sii, int stride,
                                      int x, int y, int rw, int rh)
{
    return sii[(y + rh) * stride + (x + rw)]
         - sii[y        * stride + (x + rw)]
         - sii[(y + rh) * stride +  x      ]
         + sii[y        * stride +  x      ];
}

#endif /* VJ_INTEGRAL_H */
