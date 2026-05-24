#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/vj_detect.h"
#include "../src/vj_cascade_data.h"

#define MAX_DETECTIONS 1024

static uint8_t *load_raw_image(const char *path, int expected_size)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize != expected_size) {
        fprintf(stderr, "File size %ld != expected %d\n", fsize, expected_size);
        fclose(f);
        return NULL;
    }

    uint8_t *buf = (uint8_t *)malloc(fsize);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, fsize, f);
    fclose(f);
    return buf;
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s <raw_gray_file> <width> <height> [scale_factor] [min_neighbors]\n"
        "       %s <raw_gray_file> <width> <height> [scale_factor] best\n"
        "\n"
        "  raw_gray_file : raw 8-bit grayscale image (W*H bytes, no header)\n"
        "  width height  : image dimensions\n"
        "  scale_factor  : scale step (default 1.2)\n"
        "  min_neighbors : grouping threshold (default 3, 0=raw)\n"
        "  best          : return only the single highest-confidence face\n",
        prog, prog);
}

int main(int argc, char **argv)
{
    if (argc < 4) { print_usage(argv[0]); return 1; }

    const char *img_path = argv[1];
    int w = atoi(argv[2]);
    int h = atoi(argv[3]);
    float scale_factor = (argc > 4) ? (float)atof(argv[4]) : 1.2f;
    int best_only      = (argc > 5 && strcmp(argv[5], "best") == 0);
    int min_neighbors  = (!best_only && argc > 5) ? atoi(argv[5]) : 3;

    if (w <= 0 || h <= 0) {
        fprintf(stderr, "Invalid dimensions %dx%d\n", w, h);
        return 1;
    }

    printf("Image:          %s (%dx%d)\n", img_path, w, h);
    printf("Scale factor:   %.2f\n", scale_factor);
    printf("Mode:           %s\n", best_only ? "best face only" : "all faces");
    if (!best_only) printf("Min neighbors:  %d\n", min_neighbors);
    printf("Cascade:        %d stages, %d features\n",
           vj_cascade.num_stages, vj_cascade.num_features);

    uint8_t *img = load_raw_image(img_path, w * h);
    if (!img) return 1;

    int work_size = vj_work_buf_size(w, h);
    void *work = malloc(work_size);
    if (!work) { fprintf(stderr, "Out of memory (%d bytes)\n", work_size); free(img); return 1; }

    printf("Work buffer:    %d bytes\n", work_size);
    printf("Detecting...\n");

    if (best_only) {
        vj_detection_t best;
        int found = vj_detect_best_face(&vj_cascade, img, w, h,
                                        scale_factor, &best,
                                        work, work_size);
        if (found)
            printf("Found 1 face(s)\n  [0] x=%d y=%d w=%d h=%d\n",
                   best.x, best.y, best.w, best.h);
        else
            printf("Found 0 face(s)\n");
    } else {
        vj_detection_t dets[MAX_DETECTIONS];
        int n = vj_detect_faces(&vj_cascade, img, w, h,
                                scale_factor, min_neighbors,
                                dets, MAX_DETECTIONS,
                                work, work_size);
        printf("Found %d face(s)\n", n);
        for (int i = 0; i < n; i++)
            printf("  [%d] x=%d y=%d w=%d h=%d\n",
                   i, dets[i].x, dets[i].y, dets[i].w, dets[i].h);
    }

    free(work);
    free(img);
    return 0;
}
