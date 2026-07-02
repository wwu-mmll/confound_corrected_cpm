# Design note: built-in atlas coordinates

**Status:** implemented — `develop`. Bundled atlases (16): Schaefer 100–400 (×7/×17),
Power264, Dosenbach160, Seitzman300, **Destrieux148**, **HarvardOxfordCortical**, **AAL116**,
**Glasser360**, **DesikanKilliany68**. Surface/volumetric atlases use published *volumetric
MNI* versions with computed centroids: Destrieux/Harvard-Oxford via nilearn +
`find_parcellation_cut_coords`; AAL from the SPM12 release (downloaded via curl — the host
`gin.cnrs.fr` ships a broken TLS chain that urllib/requests reject); Glasser MMP-360 from the
volumetric-MNI FSL version (`mbedini/The-HCP-MMP1.0-atlas-in-FSL`), centroids via centre-of-mass.
**Desikan-Killiany** comes from the ENIGMA Toolbox's DK labels resampled onto the conte69
(fs_LR, ≈MNI) surface — ENIGMA ships per-vertex label vectors, not centroids, so we average
vertices per parcel and assign the 68 region names by matching each parcel to the nearest
published DK centroid (Hungarian assignment, residual-asserted). All coordinate sets pass an
L/R-flip + MNI-bounding-box sanity check; Glasser's conte69-vs-volumetric centroids agree to
~3 mm. Regenerate all via `scripts/generate_atlas_data.py`.
**Author:** design draft
**Scope:** how CCCPM should provide ready-to-use brain-region names and MNI coordinates for
the HTML-report brain plots, while keeping user-supplied custom atlases working.

---

## 1. Problem

The report's brain plots (glass-brain renders, chord diagram, network-summary matrix, edge
tables) need, per connectome node:

* a **region name**, and
* **MNI coordinates** `x, y, z`,
* optionally a **network** label.

Today the user must hand this in as a CSV via `atlas_labels=<path>` with columns
`x, y, z, region` (+ optional `network`) — see
[`_validate_and_copy_atlas_file`](../src/cccpm/cpm_analysis.py) and
[`ReportDataLoader`](../src/cccpm/reporting/data_loader.py).

Producing that file is a recurring pain: coordinates are hard to obtain for a given
parcellation, and surface-based atlases (Desikan-Killiany, Destrieux, Glasser) have no native
MNI volume at all. We want the convenience of `atlas="Schaefer100"` resolving to a
ready-made, vetted coordinate table shipped with the package — without removing the
custom-file path.

---

## 2. Key finding: coordinates are computed, not looked up

There is **no public database that returns MNI centroid coordinates keyed by atlas name.**
This shapes the whole design.

* **TemplateFlow** hosts atlases as NIfTI volumes / GIFTI surfaces plus `*_dseg.tsv` label
  files. Those TSVs contain region *names and integer indices only* — **no centroids**. So
  TemplateFlow does not, by itself, solve our need.
* **Nilearn** can *fetch* many volumetric atlases (`fetch_atlas_schaefer_2018`, AAL,
  Harvard-Oxford, Yeo, …) and provides the canonical way to *derive* coordinates:
  `nilearn.plotting.find_parcellation_cut_coords(labels_img)` returns one MNI centroid per
  parcel. Coordinates are therefore **computed** from a volumetric parcellation, not read from
  a registry.
* **Exceptions** — a handful of atlases *are defined by* coordinates (spheres): Power 264,
  Dosenbach 160, Seitzman 300. Nilearn exposes these directly
  (`fetch_coords_power_2011`, `fetch_coords_dosenbach_2010`, `fetch_coords_seitzman_2018`).

Consequence: the reliable, reproducible option is to **ship precomputed, vetted coordinate
CSVs** rather than depend on runtime fetching/derivation.

---

## 3. Proposed architecture

**A single `atlas` parameter** handles both cases. The old `atlas_labels` parameter is
**removed** (deprecated with a warning first — see §3.1).

```python
CPMAnalysis(..., atlas="Schaefer100-17")     # built-in name → bundled CSV
CPMAnalysis(..., atlas="my_atlas.csv")       # custom file → loaded directly
```

Resolution rule (in `atlas`):

* If the value **points to an existing file** *or* ends with `.csv` → load it as a custom
  atlas CSV (existing validation applies).
* Otherwise → treat it as a **predefined name** and resolve via the registry.
* `None` (default) → no atlas; report falls back to atlas-free plots (as today).

Ambiguity is unlikely (a predefined name like `Schaefer100-17` won't collide with a `.csv`
path), and the rule is easy to explain. Internally both paths return the *same DataFrame*
(`region, x, y, z[, network, hemisphere, structure]`) the reporting layer already consumes —
so nothing downstream changes.

### 3.1 Deprecation of `atlas_labels`

`atlas_labels` is currently the only parameter, so we can't remove it without a transition:

* **This release:** add `atlas`; keep `atlas_labels` accepted but emit a `DeprecationWarning`
  pointing users to `atlas`. If both are given, `atlas` wins (warn about the conflict).
* **Next minor release:** remove `atlas_labels` entirely.

### Tier 1 — bundled curated CSVs (primary)

Ship vetted CSVs under `src/cccpm/atlases/`. They are a few hundred rows each (tiny), and give
us:

* **Reproducibility** — identical coordinates every run; no drift across nilearn versions.
  For a scientific package this outweighs runtime convenience.
* **Offline / CI-friendly** — no network access needed.

For most target atlases authoritative centroids are already published, so we are not inventing
numbers (see §5 for sources).

### Tier 2 — on-demand fetch + compute (fallback, deferred)

For names not bundled, optionally fetch via nilearn and run
`find_parcellation_cut_coords`, then cache to a user cache dir. Extensible, but adds an
optional nilearn dependency and requires network access. **Deferred** until a concrete need
appears — Tier 1 covers the common cases.

### Registry surface

* `cccpm.list_atlases()` → available names (discoverability).
* Name resolution is case-insensitive and tolerant of common aliases
  (`"schaefer100-17"`, `"Schaefer100-17"`).

### Naming scheme

* **Schaefer:** `Schaefer<N>-<networks>`, e.g. `Schaefer100-7`, `Schaefer100-17`,
  `Schaefer400-17`. Both 7- and 17-network parcellations are bundled for each resolution.
  No bare `Schaefer100` — the network count is always explicit to avoid an ambiguous default.
* Other atlases keep a plain descriptive name (`AAL116`, `Power264`, `DesikanKilliany`, …).

---

## 4. Surface-based atlases (Desikan-Killiany, Destrieux, Glasser)

These have no native MNI volume, so "reliable MNI coordinates" is genuinely ambiguous. Three
options, best-to-worst for our use:

1. **Ship precomputed centroids (recommended).** The **ENIGMA Toolbox** publishes centroid
   coordinate tables for `aparc` (Desikan-Killiany), `aparc.a2009s` (Destrieux), Glasser
   (HCP-MMP), and Schaefer. Bundling these treats surface atlases identically to volumetric
   ones and sidesteps the surface→volume problem entirely.
2. Use an existing **volumetric projection** (e.g. HCP-MMP1 in MNI volume; DK via FreeSurfer
   `aparc+aseg`) and compute centroids.
3. Project fsaverage vertices to MNI via **RegFusion** (Wu et al., 2018) and average per
   parcel — this is how the Schaefer centroids were made; overkill to reimplement.

Because glass-brain nodes are rendered as dots, ENIGMA's centroids are more than adequate.
**Decision: option 1.**

---

## 4a. Region grouping: functional networks + anatomical fallback

Grouping columns let the report color and partition nodes (network-summary matrix, chord
diagram, hemisphere-split layouts). The bundled schema is:

| Column | Required | Meaning |
|---|---|---|
| `region` | yes | region/parcel name |
| `x, y, z` | yes | MNI coordinates |
| `network` | optional | functional network (e.g. Schaefer 7/17 nets) |
| `hemisphere` | optional | `L` / `R` (fallback grouping) |
| `structure` | optional | `cortical` / `subcortical` (fallback grouping) |

Not every atlas ships a functional-network partition. For those (AAL, DK, Power, …) we still
want nicer plots, so **every bundled atlas gets basic anatomical grouping** — `hemisphere`
(left/right) and `structure` (cortical/subcortical) — derived deterministically:

* **`hemisphere`** — from the parcel's `x` coordinate sign (x < 0 → L, x > 0 → R), with
  midline handling, and cross-checked against the source label prefix (`lh_/rh_`, `Left-/Right-`)
  when present.
* **`structure`** — from source metadata where available (e.g. FreeSurfer `aseg` subcortical
  labels, AAL subcortical entries); default `cortical` otherwise.

Plotting precedence: use `network` when present; otherwise fall back to `structure` and/or
`hemisphere` for coloring/partitioning. This gives, at minimum, a clean hemisphere split and
cortical/subcortical separation for every atlas.

---

## 5. Candidate atlas list & coordinate sources

| Atlas (name) | Nodes | Type | Coordinate source |
|---|---|---|---|
| `Schaefer<N>-7` / `Schaefer<N>-17` (N = 100…1000) | 100–1000 | volumetric | CBIG/Yeo `..._FSLMNI152_..._Centroid_RAS.csv` (label + R/A/S ≈ MNI + network) |
| `AAL116` | 116 | volumetric | nilearn fetch + `find_parcellation_cut_coords` (baked once) |
| `Power264` | 264 | coordinate-defined | source paper / `fetch_coords_power_2011` |
| `Dosenbach160` | 160 | coordinate-defined | `fetch_coords_dosenbach_2010` |
| `Seitzman300` | 300 | coordinate-defined | `fetch_coords_seitzman_2018` |
| `DesikanKilliany` (`aparc`) | 68/84 | surface | ENIGMA Toolbox centroids |
| `Destrieux` (`aparc.a2009s`) | 148 | surface | ENIGMA Toolbox centroids |
| `Glasser` (HCP-MMP) | 360 | surface | ENIGMA Toolbox centroids |

> Node counts must be confirmed against the exact published files (e.g. DK with/without
> subcortical + brainstem). To be pinned down during implementation.

**Licensing:** these sources are openly licensed (CBIG, nilearn BSD, ENIGMA). Bundling small
derived CSVs is acceptable; record source + license + citation per file.

---

## 6. Validation: node-count check

Add a check that the resolved atlas's ROI count matches the connectome node count. A
mismatched atlas currently mislabels regions and misplaces nodes **silently** — cheap to
prevent, and it removes a nasty class of error. Applies to both `atlas=` and `atlas_labels=`.

---

## 7. Provenance & reproducibility

* Store each bundled CSV with a sibling metadata record (source URL, version, license,
  citation, generation date, and the command used if derived).
* The report should state which atlas/version was used so results are self-documenting.

---

## 8. Proposed first increment

1. Bundle: Schaefer (`-7` & `-17`, 100–400), AAL116, Power264, and DK/Destrieux/Glasser from
   ENIGMA — each with `hemisphere` + `structure` grouping columns (§4a).
2. Add the single `atlas` parameter (path-or-name resolution, §3) + `list_atlases()`,
   returning the existing report DataFrame shape.
3. Deprecate `atlas_labels` with a warning (§3.1); scheduled for removal next minor release.
4. Add the node-count validation (§6).
5. Defer the nilearn on-demand fetch tier (Tier 2) until an unbundled atlas is requested.

**Resolved decisions**

* Single `atlas` parameter for both custom files and predefined names; `atlas_labels` removed
  (deprecation window first).
* Schaefer bundles both 7 and 17 networks per resolution; explicit `Schaefer<N>-<7|17>`
  naming, no bare default.
* Every bundled atlas ships `hemisphere` (L/R) and `structure` (cortical/subcortical) so
  network-free atlases still get hemisphere-split, cortical/subcortical-aware plots.

**Open questions for review**

* Which atlases are must-haves for the first release vs. nice-to-have?
* Prefer zero new hard dependencies (bundle-only) initially? (Recommended.)
* When an atlas provides `network`, auto-enable the network-summary matrix in the report?
