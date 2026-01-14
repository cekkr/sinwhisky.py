# SinWhisky (pure C)

**SinWhisky** is a **circle-based resampler / smoother** for 1D signals (audio-like streams, sensor traces, etc.).
Given an input sequence of samples, it synthesizes intermediate samples by:

1. Fitting a **circle through each triplet** of consecutive points `(i-1, y[i-1])`, `(i, y[i])`, `(i+1, y[i+1])`.
2. Evaluating nearby circles at fractional positions.
3. **Blending** the predictions using a **sine-shaped weight** that favors the *center* of each circle.
4. Falling back to **linear interpolation** when circles are invalid (near-collinear points, numerical issues).

This repository contains a small, dependency-free **C99 + libm** implementation intended to match the behavior of the
reference Python implementation / whitepaper logic used to derive it.

---

## Features

- ✅ **Pure C (C99)** — no external dependencies besides `libm`
- ✅ **Deterministic** resampling with exact preservation of original samples
- ✅ **Aperture scaling** (optional) to stabilize circle fitting in steep segments
- ✅ **Safe fallbacks** (linear) when geometry is degenerate
- ✅ Small API: one `sw_resample_alloc()` function + parameter struct

---

## Project layout

- `sinwhisky.h` — public API
- `sinwhisky.c` — implementation
- `main.c` — small demo program (generates a sine wave and upsamples to CSV)

---

## Build

### macOS / Linux (GCC / Clang)

```bash
gcc -std=c99 -O2 sinwhisky.c main.c -lm -o sinwhisky_demo
./sinwhisky_demo 5 > out.csv
```

- The demo argument `5` means **zoom=5** (insert 5 samples between every original pair),
  i.e. an upsampling factor of **(zoom + 1) = 6x**.

### Windows (MinGW)

```bash
gcc -std=c99 -O2 sinwhisky.c main.c -lm -o sinwhisky_demo.exe
sinwhisky_demo.exe 5 > out.csv
```

For MSVC you may need to replace some `math.h` functions or ensure they are available,
and there is no `-lm` flag.

---

## Usage (as a library)

### Minimal example

```c
#include "sinwhisky.h"
#include <stdio.h>

int main(void) {
    float x[] = { 0.0f, 1.0f, 0.0f, -1.0f, 0.0f };
    size_t n_out = 0;

    sw_params_t p = sw_default_params(/*zoom=*/5);
    float* y = sw_resample_alloc(x, 5, &p, &n_out);

    if (!y) return 1;

    for (size_t i = 0; i < n_out; ++i) {
        printf("%zu %.6f\n", i, (double)y[i]);
    }

    sw_free(y);
    return 0;
}
```

### Output length rule

If the input has `n_in` samples and `zoom >= 0`, then:

```
n_out = (n_in - 1) * (zoom + 1) + 1
```

- `zoom = 0` returns the original sequence (same length).
- `zoom = 5` inserts 5 samples between each original pair (6x total rate).

---

## API reference

### `sw_params_t`

```c
typedef struct sw_params {
    int    zoom;            /* samples to INSERT between originals (>=0) */
    double collinear_eps;   /* threshold for "points nearly collinear" */

    int    use_aperture;    /* 0 disables aperture scaling */
    double aperture_gain;   /* scale factor applied to max |dy| */
    double aperture_min;    /* clamp min (recommended 1.0) */
    double aperture_max;    /* clamp max (recommended ~8..16) */
} sw_params_t;
```

**Recommended starting point:**

```c
sw_params_t p = sw_default_params(5);
```

### `sw_resample_alloc`

```c
float* sw_resample_alloc(const float* in_samples, size_t n_in,
                         const sw_params_t* params,
                         size_t* out_n);
```

- Returns a `malloc()`-allocated array of output samples (caller owns it).
- Writes output length to `out_n` (if non-NULL).
- Use `sw_free()` (or `free()`) to release the buffer.

---

## Algorithm overview (implementation notes)

### Circle fitting (local coordinates)

For each interior index `i` (`1 .. n-2`), we build the unique circle through three points.
To improve stability, we work in a local coordinate system centered at the middle point:

- `x = -aperture, 0, +aperture`
- `y = y[i-1], y[i], y[i+1]`

The circle center `(cx, cy)` is solved from a 2×2 linear system derived from the circle equation:

```
(x - cx)^2 + (y - cy)^2 = r^2
```

If the determinant is tiny (`|det| < collinear_eps`), the points are treated as collinear and the circle is invalid.

### Arc branch selection

A circle has two possible `y(x)` branches:

- `y = cy + sqrt(r^2 - (x - cx)^2)` (upper)
- `y = cy - sqrt(r^2 - (x - cx)^2)` (lower)

We pick the branch that passes through the middle sample `y[i]`.

### Blending with sine weights

Between samples `i` and `i+1`, each intermediate position `t` (0..1) can be predicted from:

- the circle centered at `i` (evaluated at `u = +t`, local right half)
- the circle centered at `i+1` (evaluated at `u = t-1`, local left half)

Each prediction gets a window weight:

- `w = sin(pi * (u+1)/2)` for `u in [-1, 1]`
- peaks at the circle center (`u = 0`)

The output is the weighted average of valid predictions.
If no valid prediction exists, we fall back to **linear interpolation**.

---

## Parameter tuning

- `collinear_eps`
  - Larger → more conservative (more linear fallbacks).
  - Smaller → more circles accepted, but potentially more numerical noise.

- `use_aperture`
  - Enable this for typical audio/sensor signals.
  - Disable only if you want the simplest geometry (x-step = 1 always).

- `aperture_gain / min / max`
  - Aperture scales the x-spacing used for circle fitting.
  - Higher values can reduce extreme radii in steep segments, but too high can flatten curvature.
  - Defaults are intended as a practical compromise.

---

## Known limitations / edge cases

- Endpoints (`i = 0` and `i = n-1`) do not have full neighborhoods; the implementation preserves them exactly
  and relies on blending/linear fallback for edge interpolation.
- Very noisy or discontinuous signals can produce circles with unrealistic curvature; aperture helps, but
  you may want to preprocess (e.g. light smoothing) for certain datasets.
- This is a **1D** method; it does not currently implement multi-channel coupling or phase-locked audio strategies.

---

## Testing tips

- Compare output against the Python reference by resampling the same `in[]` and diffing the generated CSV.
- Plot `out.csv` to visually inspect continuity and overshoot behavior.
- Stress test with:
  - constant signals
  - ramps / steps
  - alternating spikes
  - random noise

---

## License

No explicit license is included here. If you plan to publish/distribute this repository, add a license file
(e.g. MIT/BSD-2/Apache-2.0) according to your needs.

---

## Acknowledgements

This C port was derived from:
- the provided Python reference implementation (`whisky.py`)
- the original "SinWhisky e FLT" white paper

(These originals are not redistributed in this minimal C bundle.)
