# SinWhisky — Python + C implementations

SinWhisky is a **geometric, circle-based resampling / smoothing** idea:  
for any 3 consecutive samples there is exactly **one** circle passing through them; by fitting circles along a signal and blending their local arcs, you can synthesize intermediate samples and obtain a smoother, more “natural” curve than naive linear interpolation.

This repository contains:

- a **Python reference / experimental implementation** (and a circle-based audio compressor demo)
- a **portable C99 implementation** located in `sinwhisky_c/`

The original background, motivations, and the relationship to the **Fast Linear Transform (FLT)** are described in the included white paper: `SinWhisky e FLT.pdf`.

---

## Repository layout

### Python (reference + experiments)

- `whisky.py`  
  Reference utilities around **circle fitting** and signal approximation (e.g., circle-from-3-points and helpers).

- `main.py`  
  A **circle-parameter audio compressor / decompressor** demo. It reads a WAV, approximates segments using circle primitives, writes a compact binary format, and can reconstruct to WAV. It also includes analysis/visualization helpers and an optional “delta” encoding step.

> Note: `main.py` is an *experimental* compressor built on “circle primitives”.  
> The `sinwhisky_c/` code focuses on the **SinWhisky resampling** core (upsampling by inserting samples).

### C (portable SinWhisky core)

- `sinwhisky_c/`
  - `sinwhisky.h` — public API
  - `sinwhisky.c` — implementation (C99 + libm)
  - `main.c` — small demo (generates a sine wave and outputs CSV)

---

## What SinWhisky does (and doesn’t)

SinWhisky can make low-rate signals **sound/plot smoother** by inserting samples in-between known ones.  
It is **not** a perfect reconstruction method: if high-frequency content is missing in the input, it cannot be “magically” restored. In audio terms, very high frequencies tend to remain attenuated/cut when you upsample from too low a rate — the algorithm primarily improves the **perceived continuity** of the waveform.

---

## Quick start (Python)

### Requirements

- Python 3.8+ recommended
- `numpy`
- `scipy`
- (optional) `matplotlib` for visualization

Install:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy scipy matplotlib
```

### CLI usage (`main.py`)

`main.py` exposes multiple modes:

- `compress` — WAV → custom binary file
- `decompress` — custom binary → WAV
- `visualize` — plot/debug the approximation process and save figures
- `analyze` — inspect patterns in the compressed file
- `delta` — apply an extra delta-encoding step to reduce redundancy

Examples:

```bash
# Compress WAV to a binary file
python main.py --mode compress --input input.wav --output out.swk \
  --precision float16 --accuracy 0.05

# Decompress back to WAV
python main.py --mode decompress --input out.swk --output restored.wav

# Visualize compression (writes plots to a directory)
python main.py --mode visualize --input input.wav --output-dir viz --samples 2000

# Analyze the compressed file for repeated patterns / redundancy
python main.py --mode analyze --input out.swk

# Optional: delta encode a compressed file (experimental)
python main.py --mode delta --input out.swk --output out.delta
```

Key knobs:

- `--precision {float16,float32}`  
  Controls how circle parameters are serialized. `float16` is smaller but less precise.
- `--accuracy <float>`  
  Minimum approximation accuracy / stopping threshold (lower can mean more circles, larger files, slower).
- `--no-threading`  
  Disable multithreading if you want deterministic single-thread performance comparisons.
- `--debug`  
  Print verbose diagnostics.

---

## Quick start (C)

### Build the demo

```bash
cd sinwhisky_c
gcc -std=c99 -O2 sinwhisky.c main.c -lm -o sinwhisky_demo
./sinwhisky_demo 5 > out.csv
```

- `5` means `zoom=5` → insert **5** samples between each original pair (overall factor `(zoom + 1)`).
- `out.csv` can be plotted to inspect the interpolated curve.

### Use as a library

Include `sinwhisky.h` and compile `sinwhisky.c` into your project:

```c
#include "sinwhisky.h"

sw_params_t p = sw_default_params(5);
size_t n_out = 0;
float* out = sw_resample_alloc(in, n_in, &p, &n_out);
/* ... */
sw_free(out);
```

Output length rule:

```
n_out = (n_in - 1) * (zoom + 1) + 1
```

---

## How the C resampler works (high-level)

Between two known samples `i` and `i+1`, the C implementation:

1. Builds a circle for each interior sample (centered at `i`, using `i-1, i, i+1`).
2. Evaluates up to two neighboring circles at the fractional position.
3. Blends predictions using a **sine window** that favors each circle’s center.
4. Falls back to **linear interpolation** if no valid circle prediction exists (collinear points, numerical issues).

An optional “aperture” scaling modifies the local x-step before fitting circles to avoid unstable huge-radius circles in steep segments.

---

## File formats / interop notes (Python demo)

The Python compressor writes a custom binary format with:

- a header containing metadata (sample rate, original length, segment length, number of segments, normalization factor, precision code)
- per-segment data storing a sequence of circle parameter records

This format is currently **Python-only** and primarily intended for experimentation.  
If you want interop with C, treat it as a spec candidate and pin a versioned format.

---

## Roadmap / ideas

- C WAV CLI (`sinwhisky in.wav out.wav --zoom N`) for direct audio testing
- A shared, versioned binary spec for circle parameters between Python and C
- Optional “3-circle” and “N-circle” blending variants discussed in the paper (beyond immediate neighbors)
- Benchmark scripts to compare against linear/sinc interpolation on synthetic + audio datasets

---

## References

- `SinWhisky e FLT.pdf` — original white paper (SinWhisky concept + FLT discussion)

---

## License

MIT
