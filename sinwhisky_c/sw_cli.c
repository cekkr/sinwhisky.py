#include "sinwhisky.h"
#include <stdio.h>
#include <stdlib.h>

/*
  Tiny test/cross-validation CLI for SinWhisky.

  Usage:
    sw_cli <zoom> [neighbors] [use_aperture]
      zoom         : samples to insert between originals (>=0)
      neighbors    : circles per side (default 1)
      use_aperture : 1 to enable aperture scaling (default 1), 0 to disable

  Reads whitespace/newline separated floating point samples from stdin and
  prints one resampled value per line to stdout (full double precision so the
  Python reference can diff it). This keeps main.c's CSV sine demo untouched.
*/

int main(int argc, char** argv)
{
    int zoom = (argc >= 2) ? atoi(argv[1]) : 0;
    int neighbors = (argc >= 3) ? atoi(argv[2]) : 1;
    int use_aperture = (argc >= 4) ? atoi(argv[3]) : 1;
    if (zoom < 0) zoom = 0;
    if (neighbors < 1) neighbors = 1;

    size_t cap = 64, n = 0;
    float* in = (float*)malloc(cap * sizeof(float));
    if (!in) return 1;

    double v;
    while (scanf("%lf", &v) == 1) {
        if (n == cap) {
            cap *= 2;
            float* tmp = (float*)realloc(in, cap * sizeof(float));
            if (!tmp) { free(in); return 1; }
            in = tmp;
        }
        in[n++] = (float)v;
    }

    if (n == 0) { free(in); return 0; }

    sw_params_t p = sw_default_params(zoom);
    p.neighbors = neighbors;
    p.use_aperture = use_aperture;

    size_t n_out = 0;
    float* out = sw_resample_alloc(in, n, &p, &n_out);
    free(in);
    if (!out) return 2;

    for (size_t i = 0; i < n_out; ++i) {
        printf("%.17g\n", (double)out[i]);
    }

    sw_free(out);
    return 0;
}
