# SinWhisky: recovered original versus generated C

Compared on 2026-09-27:

- `original/whisky.c`: recovered 2014 implementation, `simpleWhisky()`.
- `original/flt.c`: recovered FLT equation solver.
- `sinwhisky_c/sinwhisky.c` and `sinwhisky.h`: generated portable resampler.
- `docs/SinWhisky e FLT.pdf`: particularly pages 6-8 and 9-13.
- `tests/test_sinwhisky.py`: current validation coverage.

## Conclusion

The generated implementation preserves the broad concept: construct circles
through consecutive triplets, adjust their aperture, blend overlapping
predictions with a sine-shaped window, and preserve the input samples.
It does **not** reproduce the recovered original algorithm. Its most important
change is replacing equal-angle arc subdivision with evaluation at equally
spaced x coordinates. Its aperture limits and finite-zoom blending window are
also different. These changes affect the actual waveform, not just precision.

The recovered files explicitly say they were reconstructed from printed source
with syntax and mechanical repairs. They are evidence of the 2014 implementation,
not proof of every detail of the earlier stolen version. The paper itself
distinguishes the stolen distance-weighted version from the later sine-weighted
C version. This comparison concerns the latter, represented by `original/`.

## 1. Independent check against the paper

The screenshot on page 8 supplies a complete numerical example:
input `[3, 7, 9, 4, 6]`, `zoom=3`.

| Sample position | Printed in paper | Recovered original | Generated defaults |
| --- | ---: | ---: | ---: |
| 0.00 | 3.000000 | 3.000000 | 3.000000 |
| 0.25 | 4.157316 | 4.157317 | 4.242282 |
| 0.50 | 5.213207 | 5.213208 | 5.307276 |
| 0.75 | 6.162361 | 6.162362 | 6.220486 |
| 1.00 | 7.000000 | 7.000000 | 7.000000 |
| 1.25 | 7.759608 | 7.759607 | 7.791463 |
| 1.50 | 8.408535 | 8.408533 | 8.465857 |
| 1.75 | 8.854186 | 8.854186 | 8.901041 |
| 2.00 | 9.000000 | 9.000000 | 9.000000 |
| 2.25 | 7.977124 | 7.977124 | 7.930159 |
| 2.50 | 6.500000 | 6.500000 | 6.500000 |
| 2.75 | 5.022877 | 5.022876 | 5.069841 |
| 3.00 | 4.000000 | 4.000000 | 4.000000 |
| 3.25 | 4.014061 | 4.014061 | 3.996046 |
| 3.50 | 4.369707 | 4.369707 | 4.275723 |
| 3.75 | 5.045354 | 5.045354 | 4.888831 |
| 4.00 | 6.000000 | 6.000000 | 6.000000 |

Maximum absolute difference from the printed values:

- Recovered original: approximately `1.9e-6`.
- Generated defaults: approximately `0.156523`.

This example does not activate the generated aperture clamps: the three
apertures are 6, 5, and 5 in both implementations. The difference therefore
demonstrates changes in arc sampling and blending independently of clamping.

## 2. Arc sampling: the principal conceptual difference

**Original:** `processCircle()` computes an angle for each input point, chooses
an angular direction, divides each adjacent angular interval into `zoom+1`
equal parts, and evaluates `y = oY + r*sin(theta)`.
See `original/whisky.c:186-219`, with angle helpers at lines 55-119 and 152-160.

**Generated:** `sw_circle_from_three_points()` selects the upper or lower
semicircle using the middle sample. At fractional position `g`, the evaluator
uses `x_local = (g-j)*aperture` and
`y = cy +/- sqrt(r*r - (x_local-cx)^2)`.
See `sinwhisky_c/sinwhisky.c:85-101` and 212-235.

Equal angle increments are equal arc-length increments on a given circle.
They generally do not correspond to equal x increments. The original takes
the resulting y sequence as the inserted signal samples; it does not resample
those angular points back onto a uniform geometric x grid. The generated code
implements that uniform-x interpretation directly.

For `[0, 1, 0]`, `zoom=3`, both use aperture 1 and the same circle. There is
only one circle, so blending cancels out:

| Position | Original | Generated |
| --- | ---: | ---: |
| 0.00 | 0.000000 | 0.000000 |
| 0.25 | 0.382683 | 0.661438 |
| 0.50 | 0.707107 | 0.866025 |
| 0.75 | 0.923879 | 0.968246 |
| 1.00 | 1.000000 | 1.000000 |

The original produces an angular sine arc. The generated version produces the
upper semicircle as a function of x. Both use circles, but their output
sequences are different. No existing parameter switches the generated code to
the original angular method.

## 3. Aperture: the original formula was altered

Both implementations take the maximum of **all three pairwise y differences**,
including the difference between the first and third points:

```
range = max(abs(y0-y1), abs(y1-y2), abs(y0-y2))
```

**Original:** `aperture = range`, unconditionally, with no bounds or gain.
It scales x coordinates `[0, 1, 2]` into `[0, aperture, 2*aperture]`.
See `original/whisky.c:165-173`. The code image on paper page 7 confirms this
exact formula.

**Generated defaults:** `aperture = clamp(range, 1, 12)`.
Scaling can also be disabled and the gain/bounds configured.
See `sinwhisky_c/sinwhisky.c:121-137` and 158-161.

Using `[-aperture, 0, +aperture]` instead of `[0, aperture, 2*aperture]` is
only a coordinate translation. The new gain and limits are substantive changes.

For ordinary nondegenerate triplets, the original scales both axes together
when the signal amplitude is scaled, preserving its normalized output shape
apart from floating-point effects and the absolute degeneracy threshold.
The generated limits make that shape depend on amplitude and units:

| Peak amplitude A, input `[0,A,0]` | Original y at position 0.5 / A | Generated y at position 0.5 / A |
| --- | ---: | ---: |
| 0.1 | 0.707107 | 0.751866 |
| 1 | 0.707107 | 0.866025 |
| 10 | 0.707107 | 0.866025 |
| 100 | 0.707107 | 0.996439 |

Consequently, normalizing audio before interpolation can change the generated
result even after undoing that normalization. Removing the clamps alone still
does not restore the original angular sampling.

### Branch mismatch and discontinuity

The original angular representation can traverse different halves of a circle.
The generated evaluator uses one fixed semicircle for the entire triplet. That
semicircle always contains the middle sample, but need not contain both neighbors.

A default-parameter example is `[0,100,0]`: aperture is capped at 12, so the
fitted circle has `cy=49.28` and `r=50.72`. Its middle sample is on the upper
half; both zero-valued neighbors are on the lower half. Evaluating the selected
upper half at the neighbors' x coordinates gives `98.56`, not zero.

The output overwrites the original positions with zero, but the inserted
values tend toward `98.56` as their positions approach those zeros. For
`zoom=999`, the first inserted value is approximately `98.562920`; the original
gives `0.157061`. This is a discontinuity in the interpolating prediction, not
a failure to preserve the discrete original samples.

The issue is conditional: it occurs when a triplet spans the two branches.
It can arise with capped aperture or with aperture disabled; it is not a claim
that every normalized audio input encounters it.

## 4. Blending window: same family, different finite-zoom formula

Let `q = zoom+1`. A triplet produces `2*q+1` predictions at local output
indices `j=0..2*q`.

**Original:**

```
w_old(j) = sin(pi*(j+1)/(2*q+2))
```

See `original/whisky.c:290-300`. It accumulates weighted y values and weights,
then divides by the accumulated weight.

**Generated, neighbors=1:**

```
u = (j-q)/q
w_new(u) = cos(pi*u/2)
```

See `sinwhisky_c/sinwhisky.c:21-35` and 237-246. In the same coordinates,
the original is `cos(pi*q*u/(2*(q+1)))`, which is not the generated formula
for finite q. The original has nonzero endpoint weights; the generated
window goes to zero there. They converge as zoom grows, but do not match
exactly at finite zoom.

For `zoom=3`, original weights are:

```
0.309017 0.587785 0.809017 0.951057 1 0.951057 0.809017 0.587785 0.309017
```

Generated weights at the same positions are:

```
0        0.382683 0.707107 0.923880 1 0.923880 0.707107 0.382683 0
```

Both preserve originals separately. Different relative weights at inserted
positions still change the mixture where predictions from two triplets differ.

## 5. Lines and degenerate triplets

The original tries to represent straight-line triplets as line predictions
and includes them in the same weighted blend as circle predictions.
The generated code rejects collinear circles and only uses linear interpolation
when **no** circle contributes. If another circle is valid, the rejected line
triplet contributes nothing. That changes the mixture near straight-to-curved
transitions even when the original line calculation is valid.

However, the recovered original has defects that should not be copied for the
sake of fidelity:

- `getYinLine()` returns `m*(x-xs[0])`, missing the `ys[0]` intercept
  (`original/whisky.c:42-46`). This also breaks its equality-based line detection.
- Flat signals give aperture zero, making slope and normalized-circle divisions
  degenerate (`original/whisky.c:171-178`).
- `findCircle()` sets radius zero for near-collinear points, but its caller
  continues through the angle calculation instead of taking a safe line path
  (`original/whisky.c:138-141`, 184-188).
- With two input samples no triplet is processed; the zero-initialized inserted
  samples remain zero (`original/whisky.c:250`, 267-268, 319-321).

Observed with `zoom=3`:

| Input | Recovered original behavior | Generated behavior |
| --- | --- | --- |
| `[2.5,2.5,2.5,2.5,2.5]` | Inserted values are 1 | Constant 2.5 |
| `[1,2,3,4]` | Inserted values are 1 | Exact linear ramp |
| `[1,2]` | `[1,0,0,0,2]` | `[1,1.25,1.5,1.75,2]` |

These generated behaviors are useful corrections to defects in the available
recovered source. They do not explain away the separate angular, aperture,
and window differences on valid circles.

## 6. Other changes and preserved behavior

| Aspect | Original | Generated | Assessment |
| --- | --- | --- | --- |
| Triplets | Every interior sample with its two neighbors | Same | Preserved |
| Circle fitting | Circumcircle determinant formula | Equivalent 2x2 linear system | Same geometry for identical points |
| Output length | `(n-1)*(zoom+1)+1` | Same | Preserved |
| Input sample preservation | Final overwrite | Direct copy | Preserved |
| Edge intervals | Single available triplet | Single available valid circle, or linear fallback | Same general idea; handling improved |
| Support | Each triplet covers its two adjacent intervals | Default same; configurable `neighbors>1` | Extension anticipated by paper, not in recovered code |
| Invalid prediction | Limited checking | Reject nonfinite/invalid circles and out-of-extent predictions | Robustness improvement |
| Arithmetic | Predominantly float, some double angle operations | Double geometry and accumulation; float input/output | Numerical implementation change |
| Degeneracy threshold | `1e-6` on original determinant | `1e-10` on system determinant | Different acceptance of nearly straight triplets |
| Execution | Generate triplet arrays, rolling accumulation, per-triplet allocation | Precompute circle descriptors; evaluate each output | Implementation change; no performance benchmark made |
| FLT | Separate reconstructed equation solver exists | No FLT code in `sinwhisky_c/` | Missing from generated implementation |

For identical aperture the generated system determinant is four times the
original determinant. Thus even after accounting for that factor, the default
degeneracy tolerance is much smaller. Increased precision helps, but this is
not merely the same threshold written differently.

The paper's extension to circles beyond the defining three points motivates
`neighbors>1`, but does not specify the generated generalized cosine window.
At inserted positions the recovered code normally blends two adjacent triplets.
At an interior original position up to three triplets overlap, before the
original sample is restored. The paper's wording about three circles should
not be interpreted as three contributions at every fractional position.

## 7. FLT scope

`original/flt.c` contains the model and solver described by the second half of
the paper: `generateFltTable()`, `fuseEquation2Var()`, `flt3()`, and
`confirmFlt3()`. It builds equations from measured values and the supplied
dispersion coefficients, combines equations for each unknown, substitutes
variables, and reconstructs measured values from the results.

Neither recovered file implements the full FFT calibration, frequency-band
selection, or audio equalizer pipeline. The solver's correctness and numerical
stability were not audited in this comparison. Its source also records repairs
and unresolved peculiarities, such as an unused `premolt` calculation.

The accurate status is: **a recovered FLT solver exists; the generated portable
resampler and Python reference do not integrate FLT; the complete audio pipeline
is not present.** Describing all FLT implementation as absent is now misleading.

## 8. Verification and documentation gaps

Both recovered C files and the generated demo compile without warnings using
`cc -std=c99 -O2 -Wall -Wextra -Wpedantic` (plus `-lm` for linking).
Both resamplers were compiled as shared libraries and called on identical
float32 inputs. The examples above use generated defaults and `neighbors=1`.

All **18 existing tests pass**, including C/Python cross-validation, using the
bundled Python runtime. The system `python3` cannot run them here because its
environment lacks NumPy. No dependencies were installed.

The tests compare generated C against the Python reference implementing the
same reinterpretation. They do not compare against `simpleWhisky()` or the
paper's numerical demo. In particular, the exact-circle test expects uniform-x
circle evaluation with aperture disabled; that is not the original equal-angle
sampling contract. Passing these tests establishes C/Python agreement, not
fidelity to the recovered implementation.

Repository documentation currently needs these qualifications:

- `AGENTS.md` omits `original/` and says FLT is not implemented anywhere.
- Its statement that C works internally in float is incorrect for the current
  generated core: its geometry and accumulation use double.
- The READMEs' claims of matching the paper/reference do not distinguish the
  Python/generated interpretation from the recovered angular algorithm.
- The header's edge-fallback description suggests two valid circles are
  required, but the code uses one if available.
- Extent shorthand `|x_local| <= r` in documentation omits the circle-center
  offset; the actual evaluator correctly tests `|x_local-cx| <= r`.

## 9. Suggested direction

For restoring the recovered concept, use its angular subdivision, unbounded
range-based aperture for nondegenerate triplets, and exact finite-zoom window.
Repair line offsets and degenerate cases explicitly. Preserve the generated
API, double arithmetic, finite checks, and allocation handling where useful.
Any algorithm changes must be mirrored in the Python reference as required
by the repository instructions.

Keep the current uniform-x variant available if desired, with a distinct mode
and documented semantics. Add the page 8 example and the isolated `[0,1,0]`
angular example as independent fidelity checks; retain separate coverage for
straight lines, flat signals, and branch-spanning triplets. This avoids treating
defects in the recovered source as intended behavior.

No algorithm, tests, or existing documentation were modified in this comparison.
This report is the only added repository file.
