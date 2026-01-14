\
#include "sinwhisky.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ---------- small helpers ---------- */

#ifndef SW_PI
#define SW_PI 3.141592653589793238462643383279502884
#endif

static double sw_clamp(double v, double lo, double hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static double sw_max3(double a, double b, double c) {
    double m = (a > b) ? a : b;
    return (m > c) ? m : c;
}

/* sine window on [-1, +1] -> [0, 1], peak at 0 */
static double sw_sine_weight(double u) {
    /* u in [-1,1], p in [0,1] */
    double p = (u + 1.0) * 0.5;
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return 0.0;
    return sin(SW_PI * p);
}

/* ---------- circle core ---------- */

typedef struct sw_circle {
    double cx;       /* center x in "local scaled" coordinates */
    double cy;       /* center y */
    double r;        /* radius */
    double aperture; /* scale factor applied to local x (see paper) */
    int    upper;    /* if 1 use y = cy + sqrt(...), else y = cy - sqrt(...) */
    int    valid;
} sw_circle_t;

static int sw_circle_from_three_points(double x1, double y1,
                                       double x2, double y2,
                                       double x3, double y3,
                                       double eps,
                                       sw_circle_t* out)
{
    /* Solve the 2x2 system exactly like whisky.py does via np.linalg.solve:
         [ 2(x2-x1)  2(y2-y1) ] [cx] = [x2^2-x1^2 + y2^2-y1^2]
         [ 2(x3-x2)  2(y3-y2) ] [cy]   [x3^2-x2^2 + y3^2-y2^2]
    */
    const double A00 = 2.0 * (x2 - x1);
    const double A01 = 2.0 * (y2 - y1);
    const double A10 = 2.0 * (x3 - x2);
    const double A11 = 2.0 * (y3 - y2);

    const double b0 = (x2*x2 - x1*x1) + (y2*y2 - y1*y1);
    const double b1 = (x3*x3 - x2*x2) + (y3*y3 - y2*y2);

    const double det = A00 * A11 - A01 * A10;
    if (fabs(det) < eps) {
        return 0; /* nearly collinear / unstable */
    }

    const double cx = (b0 * A11 - A01 * b1) / det;
    const double cy = (A00 * b1 - b0 * A10) / det;
    const double dx = cx - x2;
    const double dy = cy - y2;
    const double r  = sqrt(dx*dx + dy*dy);

    if (!isfinite(r) || r <= 0.0) {
        return 0;
    }

    out->cx = cx;
    out->cy = cy;
    out->r  = r;
    /* arc side: choose the branch that hits y2 at x2 */
    out->upper = (y2 >= cy) ? 1 : 0;
    out->valid = 1;
    return 1;
}

static int sw_circle_eval(const sw_circle_t* c, double x_local, double* y_out)
{
    /* circle: (x - cx)^2 + (y - cy)^2 = r^2 */
    const double dx = x_local - c->cx;
    const double inside = c->r * c->r - dx * dx;
    if (inside < 0.0) return 0;
    const double dy = sqrt(inside);
    *y_out = c->upper ? (c->cy + dy) : (c->cy - dy);
    return isfinite(*y_out) ? 1 : 0;
}

/* Build circle centered at index i using samples (i-1, i, i+1) in local coords. */
static sw_circle_t sw_build_circle(const float* y, size_t n, size_t i,
                                  const sw_params_t* p)
{
    sw_circle_t c;
    memset(&c, 0, sizeof(c));
    c.valid = 0;

    if (i == 0 || i + 1 >= n) return c;

    const double y0 = (double)y[i - 1];
    const double y1 = (double)y[i];
    const double y2 = (double)y[i + 1];

    double aperture = 1.0;
    if (p->use_aperture) {
        /* "apertura" idea from the paper: scale the x-step using max |dy| */
        const double d01 = fabs(y0 - y1);
        const double d12 = fabs(y1 - y2);
        const double d02 = fabs(y0 - y2);
        const double maxapr = sw_max3(d01, d12, d02);
        aperture = sw_clamp(maxapr * p->aperture_gain, p->aperture_min, p->aperture_max);
    }

    /* local coordinates around the middle sample:
         x = -aperture, 0, +aperture
       This matches the "increment = 1, then temporarily change it" description.
    */
    const double x0 = -aperture;
    const double x1 = 0.0;
    const double x2 = +aperture;

    c.aperture = aperture;

    if (!sw_circle_from_three_points(x0, y0, x1, y1, x2, y2, p->collinear_eps, &c)) {
        c.valid = 0;
        return c;
    }
    return c;
}

/* ---------- public API ---------- */

sw_params_t sw_default_params(int zoom)
{
    sw_params_t p;
    p.zoom = (zoom < 0) ? 0 : zoom;
    p.collinear_eps = 1e-10; /* matches whisky.py */

    p.use_aperture  = 1;
    p.aperture_gain = 1.0;
    p.aperture_min  = 1.0;
    p.aperture_max  = 12.0;
    return p;
}

void sw_free(void* p) { free(p); }

float* sw_resample_alloc(const float* in_samples, size_t n_in,
                         const sw_params_t* params,
                         size_t* out_n)
{
    if (!in_samples || n_in == 0) return NULL;

    sw_params_t p = params ? *params : sw_default_params(0);
    if (p.zoom < 0) p.zoom = 0;

    if (n_in == 1) {
        float* out = (float*)malloc(sizeof(float));
        if (!out) return NULL;
        out[0] = in_samples[0];
        if (out_n) *out_n = 1;
        return out;
    }

    const size_t zoom = (size_t)p.zoom;
    const size_t n_out = (n_in - 1) * (zoom + 1) + 1;

    float* out = (float*)malloc(n_out * sizeof(float));
    if (!out) return NULL;

    /* Precompute circles for centers 1..n_in-2 */
    sw_circle_t* circles = (sw_circle_t*)malloc(n_in * sizeof(sw_circle_t));
    if (!circles) { free(out); return NULL; }
    for (size_t i = 0; i < n_in; ++i) {
        circles[i].valid = 0;
    }
    for (size_t i = 1; i + 1 < n_in; ++i) {
        circles[i] = sw_build_circle(in_samples, n_in, i, &p);
    }

    size_t w = 0;
    for (size_t i = 0; i + 1 < n_in; ++i) {
        /* write original sample */
        out[w++] = in_samples[i];

        /* insert zoom samples between i and i+1 */
        for (size_t k = 1; k <= zoom; ++k) {
            const double t = (double)k / (double)(zoom + 1); /* in (0,1) */

            double sum_w = 0.0;
            double sum_y = 0.0;

            /* left circle centered at i (needs i in [1..n-2]) */
            if (i >= 1 && i + 1 < n_in && circles[i].valid) {
                const double u = +t; /* local u in [0,1] */
                const double x_local = u * circles[i].aperture;
                double y_pred;
                if (sw_circle_eval(&circles[i], x_local, &y_pred)) {
                    const double ww = sw_sine_weight(u);
                    sum_w += ww;
                    sum_y += ww * y_pred;
                }
            }

            /* right circle centered at i+1 */
            if (i + 1 >= 1 && i + 2 < n_in && circles[i+1].valid) {
                const double u = t - 1.0; /* local u in [-1,0] */
                const double x_local = u * circles[i+1].aperture;
                double y_pred;
                if (sw_circle_eval(&circles[i+1], x_local, &y_pred)) {
                    const double ww = sw_sine_weight(u);
                    sum_w += ww;
                    sum_y += ww * y_pred;
                }
            }

            double y_out;
            if (sum_w > 1e-20) {
                y_out = sum_y / sum_w;
            } else {
                /* fallback: linear interpolation */
                const double y0 = (double)in_samples[i];
                const double y1 = (double)in_samples[i + 1];
                y_out = (1.0 - t) * y0 + t * y1;
            }

            out[w++] = (float)y_out;
        }
    }

    /* last original */
    out[w++] = in_samples[n_in - 1];

    free(circles);

    if (out_n) *out_n = n_out;
    return out;
}
