\
#include "sinwhisky.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Demo:
   - generate a 1 kHz sine sampled at 8 kHz for 20 ms
   - upsample with SinWhisky (zoom=5 => 6x -> 48 kHz)
   - write CSV to stdout: t_in_seconds, y
*/

int main(int argc, char** argv)
{
    int zoom = 5;
    if (argc >= 2) zoom = atoi(argv[1]);
    if (zoom < 0) zoom = 0;

    const double sr_in  = 8000.0;
    const double freq   = 1000.0;
    const double dur_s  = 0.020; /* 20 ms */
    const size_t n_in   = (size_t)ceil(sr_in * dur_s);

    float* in = (float*)malloc(n_in * sizeof(float));
    if (!in) return 1;

    for (size_t i = 0; i < n_in; ++i) {
        double t = (double)i / sr_in;
        in[i] = (float)sin(2.0 * 3.14159265358979323846 * freq * t);
    }

    sw_params_t p = sw_default_params(zoom);
    size_t n_out = 0;
    float* out = sw_resample_alloc(in, n_in, &p, &n_out);
    free(in);
    if (!out) return 2;

    const double sr_out = sr_in * (double)(zoom + 1);

    /* CSV header */
    printf("t,y\n");
    for (size_t i = 0; i < n_out; ++i) {
        double t = (double)i / sr_out;
        printf("%.9f,%.9f\n", t, (double)out[i]);
    }

    sw_free(out);
    return 0;
}
