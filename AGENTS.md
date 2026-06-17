# AGENTS.md — notes for AI assistants working on SinWhisky

Orientation file for future agents. Read this first, then the relevant
`README.md`. Keep it updated when you change the algorithm, the layout, or the
build/test story.

---

## 1. What this project is

**SinWhisky** is a *circle-based resampler / smoother* for 1‑D signals (audio,
sensor traces). Core idea from the whitepaper: through 2 points pass infinite
circles, through 3 points exactly one. For every interior sample fit the unique
circle through it and its two neighbours, then synthesise the in‑between samples
by blending the nearby circles with a window that favours each circle's centre.

It is **not** perfect reconstruction: missing high frequencies are not restored,
it just makes a low‑rate signal *look/sound* more continuous.

The whitepaper also sketches a second, separate idea — **FLT (Fast Linear
Transform)**, a linear‑system equalizer to undo an unknown chip's distortion.
**FLT is described in the paper but NOT implemented in this repo yet.**

Source of truth: `docs/SinWhisky e FLT.pdf` (Italian, R. Cecchini, 2014).

---

## 2. Repository structure

```
.
├── docs/SinWhisky e FLT.pdf   # whitepaper (Italian). 13 pages; algo on p.6–8, FLT p.9–12
├── README.md                  # top-level overview
├── main.py                    # experimental WAV compressor (circle primitives) — OUT OF SCOPE for the resampler
├── v2/whisky.py               # Python reference: sinwhisky_resample() + circle utils + compressor
├── sinwhisky_c/               # C99 resampler core (the portable implementation)
│   ├── sinwhisky.h            #   public API + sw_params_t
│   ├── sinwhisky.c            #   implementation
│   ├── main.c                 #   CSV sine demo: ./sinwhisky_demo <zoom>
│   ├── sw_cli.c               #   stdin CLI used for C↔Python cross-validation
│   └── README.md
├── tests/test_sinwhisky.py    # unittest suite (geometry → invariants → analytic → behaviour → C/Py cross-check → compressor)
└── test.sh                    # ad-hoc main.py compress/decompress invocation (needs a chopin.wav)
```

Two implementations matter and **must stay consistent**:
`sinwhisky_c/` (C, portable core) ↔ `v2/whisky.py::sinwhisky_resample` (Python
reference). The test suite enforces this. `main.py` is a separate experiment.

---

## 3. The algorithm contract (keep C and Python identical)

When changing the resampler, change **both** `sinwhisky.c` and `whisky.py` the
same way, then run the cross-validation test.

- Output length: `n_out = (n-1)*(zoom+1) + 1`. Original samples preserved exactly
  at indices that are multiples of `(zoom+1)`.
- Per interior index `i` (1..n-2): aperture
  `a = clamp(max(|Δy|)*gain, amin, amax)` (or `1.0` if aperture disabled). Fit the
  circle through `(-a, y[i-1]), (0, y[i]), (+a, y[i+1])`. Branch = upper iff
  `y[i] ≥ cy`. The branch is single-valued/continuous over `(cx-r, cx+r)`, so it
  stays correct even when extrapolated.
- Output at global position `g = i + t` (`t = k/(zoom+1)`): blend every valid
  interior circle `j` with `|g-j| ≤ neighbors`, evaluated at `u = g-j`,
  `x_local = u·a_j`, weight `w(u) = cos(π·u/(2·neighbors))`. Skip a circle if
  `|x_local| > r` (outside its horizontal extent). Fall back to **linear
  interpolation** when no circle contributes.
- `neighbors = 1` (default) reproduces the classic two-circle blend, because
  `cos(πu/2) == sin(π(u+1)/2)`. `neighbors > 1` is the paper's "circles beyond the
  three circumscribed points" extension.

### Gotchas / things that bit us
- **Aperture distorts exact-circle reproduction.** Rescaling x makes the fitted
  curve an ellipse in real coordinates. Tests that expect points on a true circle
  to be recovered set `use_aperture=False`.
- **Precision:** C works in `float` (32‑bit) internally and casts output to
  `float`. The Python reference is full `float64`. Cross-validation therefore uses
  `atol≈2e-4`, not exact equality.
- **Accumulation order** is `j` ascending in both implementations — keep it that
  way so the blend matches.
- The aperture formula is deliberately simple ("perfectible" per the paper). Don't
  assume it's optimal.

---

## 4. Build & test

```bash
# C demo + cross-validation CLI (clang/gcc/cc all work; macOS arm64 OK)
cd sinwhisky_c
gcc -std=c99 -O2 -Wall -Wextra -Wpedantic sinwhisky.c main.c   -lm -o sinwhisky_demo
gcc -std=c99 -O2 -Wall -Wextra -Wpedantic sinwhisky.c sw_cli.c -lm -o sw_cli
echo "0 1 0 -1 0" | ./sw_cli 3 1 1     # sw_cli <zoom> [neighbors] [use_aperture]

# Tests (compiles sw_cli automatically; skips the C cross-check if no compiler)
python3 -m unittest discover -s tests -v
```

Keep the C build warning-clean under `-Wall -Wextra -Wpedantic`.

---

## 5. Environment notes (this machine, may differ elsewhere)

- Use `python3`. There is **no `pip`** on PATH — use `python3 -m pip`.
- `numpy` is installed; **`scipy` and `matplotlib` are NOT.** Keep `v2/whisky.py`
  importable without them (matplotlib is already imported lazily). Don't add hard
  deps on scipy/matplotlib to importable code paths.
- No `pdftotext`/`poppler`/`mutool`. To read the PDF, `python3 -m pip install
  pypdf` then `PdfReader(...).pages[i].extract_text()`. The repo's own `Read` of a
  PDF needs poppler, which isn't installed.
- Compiled binaries (`sinwhisky_demo`, `sw_cli`, `*.o`, `*.out`) are gitignored.

---

## 6. Roadmap

Done:
- ✅ Portable C99 resampler with aperture + sine-weighted blend.
- ✅ Python reference (`sinwhisky_resample`) kept consistent with C.
- ✅ Optional N-circle blending via the `neighbors` parameter (paper p.7).
- ✅ Unit tests incl. C↔Python cross-validation.

Open ideas (in rough priority order):
- **C WAV CLI** (`sinwhisky in.wav out.wav --zoom N`) for direct audio testing.
- **Better blending / aperture formula** — the paper calls both "perfectible".
  Candidates: distance-and-curvature aware aperture; smarter overlap than a plain
  weighted average.
- **Implement FLT** (the paper's second half): pick a frequency band, FFT a fixed
  window, build the linear system `V = A·x` (measured spectra vs clean-tone
  dispersion), solve robustly (the paper sums each unknown's equations; least
  squares is the modern choice), turn the result into an equalizer.
- Shared, versioned binary spec for circle parameters between Python and C.
- Benchmark scripts vs linear/sinc interpolation on synthetic + audio datasets.
- Reconcile `main.py`'s compressor with `v2/whisky.py`'s circle utilities (today
  they're separate experiments).

---

## 7. Conventions

- Comments/docstrings in `v2/whisky.py` are in Italian (matches the paper/author).
  Match the surrounding language when editing a file.
- License is MIT (`LICENSE` at repo root).
- When you touch the algorithm, update: the C source, `v2/whisky.py`, the tests,
  the two `README.md` files, and this file's roadmap.
