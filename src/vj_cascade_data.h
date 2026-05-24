#ifndef VJ_CASCADE_DATA_H
#define VJ_CASCADE_DATA_H

#include "vj_types.h"

#define VJ_NUM_FEATURES 2913
#define VJ_NUM_STAGES 25
#define VJ_NUM_WEAK_CLASSIFIERS 2913
#define VJ_WINDOW_W 24
#define VJ_WINDOW_H 24

extern const vj_feature_t vj_features[2913];
extern const vj_weak_classifier_t vj_weak_classifiers[2913];
extern const vj_stage_t vj_stages[25];
extern const vj_cascade_t vj_cascade;

#endif /* VJ_CASCADE_DATA_H */
