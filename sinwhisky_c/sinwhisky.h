\
#ifndef SINWHISKY_H
#define SINWHISKY_H

/*
  SinWhisky (C) — circle-based resampling / smoothing.

  Core idea (white paper):
    - for each sample i (except endpoints) build the unique circle through
      (i-1, y[i-1]), (i, y[i]), (i+1, y[i+1]);
    - optionally adjust the x-step ("aperture") before circle fitting to avoid
      extremely large radii when y varies a lot;
    - when synthesizing intermediate samples, blend the nearby circles
      using a sine weight that favors the center of each circle.

  This implementation stays dependency-free (C99 + libm).
*/

#include <stddef.h> /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sw_params {
    int    zoom;            /* number of samples to INSERT between originals (>=0) */
    double collinear_eps;   /* threshold for "points nearly collinear" */
    /* "Aperture" (x scaling) as described in the paper (page 7). */
    int    use_aperture;    /* 0 disables aperture scaling */
    double aperture_gain;   /* multiply max |dy| by this */
    double aperture_min;    /* clamp min (recommended 1.0) */
    double aperture_max;    /* clamp max (recommended ~8..16) */
} sw_params_t;

/* Returns sensible defaults. */
sw_params_t sw_default_params(int zoom);

/*
  Resample/upsample a 1D signal.

  Input:
    - in_samples: array of floats length n_in
    - params.zoom decides output length:
        n_out = (n_in - 1) * (zoom + 1) + 1
  Output:
    - returns malloc()'d float array, or NULL on failure
    - out_n receives output length (may be NULL)

  Notes:
    - End segments that cannot be covered by two valid circles fall back to
      linear interpolation.
    - Original samples are preserved exactly (at integer positions).
*/
float* sw_resample_alloc(const float* in_samples, size_t n_in,
                         const sw_params_t* params,
                         size_t* out_n);

/* Convenience: free() the pointer returned by sw_resample_alloc. */
void sw_free(void* p);

#ifdef __cplusplus
}
#endif

#endif /* SINWHISKY_H */
