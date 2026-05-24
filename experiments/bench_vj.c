/*
 * bench_vj.c — Viola-Jones latency + cascade statistics benchmark.
 *
 * Outputs two sections:
 *   1. Timing: mean/std/min/max over N_RUNS calls to vj_detect_faces().
 *   2. Stage stats: per-stage rejection counts from ONE instrumented pass.
 *      The instrumented eval is a local copy of src/vj_detect.c's inner
 *      loop with counters added; it does NOT affect production src/.
 *
 * Build: see build_bench.bat
 * Usage: bench_vj.exe <raw_gray> <w> <h> <scale_factor> <min_neighbors> [n_runs]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../src/vj_detect.h"
#include "../src/vj_integral.h"
#include "../src/vj_cascade_data.h"

#define MAX_DET      1024
#define MAX_RUNS     1000
#define MAX_STAGES   32

static double query_ms(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC * 1000.0;
}

/* ── Stage rejection instrumentation ───────────────────────────────────────
 *
 * This is a LOCAL COPY of the cascade evaluation logic from
 * src/vj_detect.c's vj_evaluate_window(), augmented with per-stage
 * rejection counters.  Production detection (timing benchmark above)
 * still uses the unmodified src/ function.  Any divergence between
 * the two would indicate a copy error, not a bug in src/.
 */

static long long g_stage_entered [MAX_STAGES];
static long long g_stage_rejected[MAX_STAGES];
static long long g_total_feats;   /* total weak-classifier evaluations */

static void eval_window_stats(const vj_cascade_t *cas,
                               const uint32_t *ii, const uint64_t *sii,
                               int ii_stride,
                               int win_x, int win_y,
                               int win_w, int win_h)
{
    float inv_area = 1.0f / (float)(win_w * win_h);
    uint32_t s_sum  = vj_rect_sum   (ii,  ii_stride, win_x, win_y, win_w, win_h);
    uint64_t s_sq   = vj_rect_sum_sq(sii, ii_stride, win_x, win_y, win_w, win_h);
    float    mean   = (float)s_sum * inv_area;
    float    var    = (float)s_sq  * inv_area - mean * mean;
    float    stdev  = (var > 0.0f) ? sqrtf(var) : 1.0f;
    float    scx    = (float)win_w / (float)cas->window_w;
    float    scy    = (float)win_h / (float)cas->window_h;

    for (int s = 0; s < cas->num_stages; s++) {
        g_stage_entered[s]++;
        const vj_stage_t *stage = &cas->stages[s];
        float stage_sum = 0.0f;

        for (int w = 0; w < stage->num_weak; w++) {
            const vj_weak_classifier_t *wc =
                &cas->weak_classifiers[stage->weak_start_idx + w];
            const vj_feature_t *feat = &cas->features[wc->feature_idx];
            float feat_val = 0.0f;
            for (int r = 0; r < feat->num_rects; r++) {
                const vj_rect_t *rc = &feat->rects[r];
                int rx = win_x + (int)(rc->x * scx);
                int ry = win_y + (int)(rc->y * scy);
                int rw = (int)(rc->w * scx); if (rw < 1) rw = 1;
                int rh = (int)(rc->h * scy); if (rh < 1) rh = 1;
                feat_val += (float)vj_rect_sum(ii, ii_stride, rx, ry, rw, rh)
                            * rc->weight;
            }
            feat_val *= inv_area;
            stage_sum += (feat_val < wc->threshold * stdev)
                         ? wc->left_val : wc->right_val;
        }
        g_total_feats += stage->num_weak;

        if (stage_sum < stage->stage_threshold) {
            g_stage_rejected[s]++;
            return;
        }
    }
}

static void run_stats_pass(const uint8_t *img, int img_w, int img_h, float sf,
                            uint32_t *ii, uint64_t *sii)
{
    memset(g_stage_entered,  0, sizeof(g_stage_entered));
    memset(g_stage_rejected, 0, sizeof(g_stage_rejected));
    g_total_feats = 0;

    int ii_stride = img_w + 1;
    vj_compute_integral_images(img, img_w, img_h, ii, sii);

    float scale = 1.0f;
    while (1) {
        int ww = (int)(vj_cascade.window_w * scale);
        int wh = (int)(vj_cascade.window_h * scale);
        if (ww > img_w || wh > img_h) break;
        int step = (int)(scale * 2.0f + 0.5f);
        if (step < 1) step = 1;
        for (int y = 0; y + wh <= img_h; y += step)
            for (int x = 0; x + ww <= img_w; x += step)
                eval_window_stats(&vj_cascade, ii, sii, ii_stride,
                                  x, y, ww, wh);
        scale *= sf;
    }
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr,
            "Usage: %s <raw_gray> <w> <h> <scale_factor> <min_neighbors> [n_runs]\n",
            argv[0]);
        return 1;
    }

    const char *path = argv[1];
    int   img_w    = atoi(argv[2]);
    int   img_h    = atoi(argv[3]);
    float sf       = (float)atof(argv[4]);
    int   best_mode = (strcmp(argv[5], "best") == 0);
    int   mn        = best_mode ? 0 : atoi(argv[5]);
    int   n_runs   = (argc > 6) ? atoi(argv[6]) : 50;
    if (n_runs < 1)  n_runs = 1;
    if (n_runs > MAX_RUNS) n_runs = MAX_RUNS;

    /* Load image */
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); return 1; }
    int sz = img_w * img_h;
    uint8_t *img = (uint8_t *)malloc(sz);
    if (!img) { fclose(fp); return 1; }
    fread(img, 1, sz, fp);
    fclose(fp);

    /* Work buffer (reused for both timing and stats) */
    int work_sz = vj_work_buf_size(img_w, img_h);
    void *work = malloc(work_sz);
    if (!work) { free(img); return 1; }

    int padded = (img_w + 1) * (img_h + 1);
    uint32_t *ii  = (uint32_t *)work;
    uint64_t *sii = (uint64_t *)((uint8_t *)work + padded * (int)sizeof(uint32_t));

    vj_detection_t dets[MAX_DET];

    /* ── Warm-up ── */
    int n_det = 0;
    vj_detection_t best_det;
    for (int i = 0; i < 3; i++) {
        if (best_mode)
            n_det = vj_detect_best_face(&vj_cascade, img, img_w, img_h,
                                        sf, &best_det, work, work_sz);
        else
            n_det = vj_detect_faces(&vj_cascade, img, img_w, img_h,
                                    sf, mn, dets, MAX_DET, work, work_sz);
    }

    /* ── Timing benchmark ── */
    double times[MAX_RUNS];
    for (int i = 0; i < n_runs; i++) {
        double t0 = query_ms();
        if (best_mode)
            n_det = vj_detect_best_face(&vj_cascade, img, img_w, img_h,
                                        sf, &best_det, work, work_sz);
        else
            n_det = vj_detect_faces(&vj_cascade, img, img_w, img_h,
                                    sf, mn, dets, MAX_DET, work, work_sz);
        times[i] = query_ms() - t0;
    }

    double tsum = 0.0, tsum2 = 0.0;
    double tmin = times[0], tmax = times[0];
    for (int i = 0; i < n_runs; i++) {
        tsum  += times[i];
        tsum2 += times[i] * times[i];
        if (times[i] < tmin) tmin = times[i];
        if (times[i] > tmax) tmax = times[i];
    }
    double mean = tsum / n_runs;
    double var  = tsum2 / n_runs - mean * mean;
    double std  = (var > 0.0) ? sqrt(var) : 0.0;

    /* ── Window candidate count ── */
    long long win_count = 0;
    {
        float sc = 1.0f;
        while (1) {
            int ww = (int)(vj_cascade.window_w * sc);
            int wh = (int)(vj_cascade.window_h * sc);
            if (ww > img_w || wh > img_h) break;
            int step = (int)(sc * 2.0f + 0.5f); if (step < 1) step = 1;
            long long nx = (img_w - ww) / step + 1;
            long long ny = (img_h - wh) / step + 1;
            if (nx > 0 && ny > 0) win_count += nx * ny;
            sc *= sf;
        }
    }

    /* ── Stage rejection stats (single instrumented pass) ── */
    run_stats_pass(img, img_w, img_h, sf, ii, sii);

    /* Total weak classifiers per stage for effective-feature calculation */
    long long max_feats_per_window = 0;
    for (int s = 0; s < vj_cascade.num_stages; s++)
        max_feats_per_window += vj_cascade.stages[s].num_weak;

    /* ── Print results ── */
    printf("detections=%d\n",            n_det);
    printf("n_runs=%d\n",                n_runs);
    printf("mean_ms=%.4f\n",             mean);
    printf("std_ms=%.4f\n",              std);
    printf("min_ms=%.4f\n",              tmin);
    printf("max_ms=%.4f\n",              tmax);
    printf("window_candidates=%lld\n",   win_count);
    printf("max_feats_per_window=%lld\n",max_feats_per_window);
    printf("total_feats_evaluated=%lld\n",g_total_feats);
    if (win_count > 0)
        printf("avg_feats_per_window=%.2f\n",
               (double)g_total_feats / (double)win_count);

    printf("num_stages=%d\n", vj_cascade.num_stages);
    for (int s = 0; s < vj_cascade.num_stages; s++) {
        long long entered  = g_stage_entered[s];
        long long rejected = g_stage_rejected[s];
        double rej_pct = entered > 0
                         ? 100.0 * (double)rejected / (double)entered
                         : 0.0;
        printf("stage_%d_entered=%lld\n",    s, entered);
        printf("stage_%d_rejected=%lld\n",   s, rejected);
        printf("stage_%d_rej_pct=%.2f\n",    s, rej_pct);
        printf("stage_%d_num_weak=%d\n",     s, vj_cascade.stages[s].num_weak);
    }

    free(img);
    free(work);
    return 0;
}
