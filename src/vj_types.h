#ifndef VJ_TYPES_H
#define VJ_TYPES_H

#include <stdint.h>

#define VJ_MAX_RECTS 3

typedef struct {
    int16_t x, y, w, h;
    float weight;
} vj_rect_t;

typedef struct {
    vj_rect_t rects[VJ_MAX_RECTS];
    uint8_t num_rects;
} vj_feature_t;

typedef struct {
    uint16_t feature_idx;
    float threshold;
    float left_val;
    float right_val;
} vj_weak_classifier_t;

typedef struct {
    uint16_t num_weak;
    uint16_t weak_start_idx;
    float stage_threshold;
} vj_stage_t;

typedef struct {
    int window_w;
    int window_h;
    int num_stages;
    int num_features;
    int num_weak_total;
    const vj_stage_t *stages;
    const vj_weak_classifier_t *weak_classifiers;
    const vj_feature_t *features;
} vj_cascade_t;

typedef struct {
    int16_t x, y, w, h;
} vj_detection_t;

#endif /* VJ_TYPES_H */
