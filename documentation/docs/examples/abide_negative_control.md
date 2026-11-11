# ABIDE Negative Control: Does Confound Correction Fix a Genuinely Wrong Result?

The [regression](regression.md) and [classification](classification.md) examples
run on simulated data with a *known* amount of confounding, so you can check the
correction against ground truth. On real data, that check is usually impossible:
if a naive and a confound-corrected analysis give different numbers, you cannot
tell how much of the difference is confound removed and how much is signal lost.

This example builds a real-data case where the ground truth is known anyway. The
script below is the runnable file `examples/abide_negative_control_cpm.py`.

## The full script

```python
--8<-- "examples/abide_negative_control_cpm.py"
```

## The idea

[ABIDE I](https://fcon_1000.projects.nitrc.org/indi/abide/) pools resting-state
fMRI from ~20 independent scan sites. Site is a textbook confound candidate here:
different scanners and protocols leave a signature in the connectome, and the
sites recruited different age ranges, so site correlates with both connectome
*and* age.

The script exploits that structure to build a target with a **known** effect of
exactly zero:

1. Take the real age of each of the 871 subjects.
2. **Permute it within each site** — shuffle ages among subjects *at the same
   site* only, never across sites.

This single step does two things at once:

- It destroys any individual relationship between a subject's connectome and
  their (now shuffled) age — nobody's brain data is paired with their own age
  anymore, so the true connectome-age effect is exactly 0 *by construction*.
- It leaves each site's **mean** age unchanged (the script asserts this), so the
  site-age association survives untouched — and the site-connectome association
  was never touched at all.

The confounding path `age ← site → connectome` is still fully intact; only the
direct, individual-level `connectome → age` link is gone. Any association a CPM
analysis finds between connectome and this permuted age is necessarily a false
positive manufactured through site, not a true effect — because there is no true
effect to find.

## What to expect

Two `CPMAnalysis` runs on the same permuted target, one without site as a
covariate and one with it (`selection_input` and `model_input` both
`"residualized"` — see [Confound control](../methods.md#2-confound-control)):

| | naive | site-corrected |
|---|---|---|
| r (predicted vs. permuted age) | ≈ +0.25 | ≈ −0.06 |
| permutation *p* | ≈ 0.001 (significant) | ≈ 0.90 (not significant) |

The naive run reports a confident, "significant" connectome-age relationship —
a pure false positive, since the ground-truth effect is zero. The corrected run
correctly finds nothing. Because the ground truth here is known rather than
assumed, this is evidence that confound correction fixes a genuinely wrong
result, not just that it changes a number.

!!! note "Why this is stronger than it looks"
    With real, unpermuted data you can never fully separate "the correction
    removed noise" from "the correction removed signal" — you don't know which
    one is true. Here you do, because the individual-level effect was set to
    zero by the permutation itself.

## Caveats

- This only demonstrates the extreme case (100% confound, 0% true signal). For
  the harder, more realistic case — a real effect that is *partially* confounded
  — see `examples/abide_age_site_cpm.py`, which runs the same site correction on
  the real (unpermuted) age and shows the effect shrink rather than vanish.
- No atlas is passed to `CPMAnalysis` here, so the report has no brain figures —
  edges are only identifiable by their numeric CC200 index, not by anatomical
  region.
- `fetch_abide_pcp` downloads several hundred MB on first use (cached under
  `~/nilearn_data/`), and the full run (1000 permutations × 2 analyses) takes a
  few minutes on CPU.

## See also

- [How CCCPM Works § Confound control](../methods.md#2-confound-control) — what
  `selection_input` and `model_input` each do, and why they are independent.
- [Regression example](regression.md) — the same workflow on simulated data with
  analytically known confounding.
- [Interpreting Results](../interpreting_results.md) — what every report section
  means.
