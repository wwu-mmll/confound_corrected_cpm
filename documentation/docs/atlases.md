# Brain Atlases

CCCPM works on connectomes, so the nodes of your `X` matrix correspond to regions of a brain
parcellation. Providing an **atlas** — region names and their MNI coordinates — lets the HTML
report label edges and draw the brain figures in the **Brain & Edges** section: the
connectivity matrix, hub plot, and (with coordinates) a glass-brain render, plus a
network-summary matrix and chord diagram when the atlas defines functional networks.

The atlas is **optional**. Without one, the analysis runs exactly the same and the report still
shows the connectivity matrix and hub plot — only the region names and coordinate-based figures
are omitted.

You supply it through a single `atlas` argument, which accepts **either a built-in atlas name
or a path to your own CSV file**:

```python
from cccpm import CPMAnalysis

# a) a bundled atlas, selected by name
cpm = CPMAnalysis(results_directory="results/", atlas="Schaefer100-17")

# b) your own atlas file
cpm = CPMAnalysis(results_directory="results/", atlas="my_atlas.csv")
```

If the value points to an existing file or ends in `.csv`, it is loaded as a custom atlas;
otherwise it is looked up in the built-in registry (case-insensitive). The resolved atlas is
copied into `results/edges/atlas.csv` so the report is self-contained.

!!! warning "The atlas must match your connectome"
    The number of regions in the atlas must equal the number of nodes implied by your `X`
    matrix (for an upper-triangle connectome, `n_features = n_nodes * (n_nodes - 1) / 2`).
    CCCPM checks this and raises an error if they disagree, so a mismatched atlas can never
    silently mislabel your regions.

---

## Built-in atlases

List the available names from Python:

```python
from cccpm.atlases import list_atlases
print(list_atlases())
```

| Atlas name | Nodes | Networks | Coverage |
|---|---|:---:|---|
| `Schaefer100-7` … `Schaefer400-7` | 100–400 | ✅ (7) | cortical |
| `Schaefer100-17` … `Schaefer400-17` | 100–400 | ✅ (17) | cortical |
| `Power264` | 264 | — | cortical |
| `Dosenbach160` | 160 | ✅ | cortical + cerebellum |
| `Seitzman300` | 300 | ✅ | whole-brain |
| `AAL116` | 116 | — | whole-brain |
| `HarvardOxfordCortical` | 96 | — | cortical |
| `Destrieux148` | 148 | — | cortical |
| `Glasser360` | 360 | — | cortical |
| `DesikanKilliany68` | 68 | — | cortical |

Schaefer comes in 7- and 17-network variants at 100/200/300/400 parcels; the name is always
`Schaefer<parcels>-<networks>`, e.g. `Schaefer200-17`. Atlases marked **Networks ✅** carry a
functional-network label per region, which automatically enables the network-summary matrix
and chord diagram in the report.

All coordinates are in MNI space (mm). Each atlas is derived from an authoritative public
source; the exact source and citation for every atlas are listed in
[`SOURCES.md`](https://github.com/wwu-mmll/confound_corrected_cpm/blob/develop/src/cccpm/atlases/data/SOURCES.md).
Please cite the original atlas paper when you use one.

!!! note "Coordinate space and centroids"
    Coordinates are region **centroids** — one representative point per region, intended for
    placing nodes in the brain figures, not as seed coordinates for ROI analysis. The bundled
    atlases live in slightly different MNI-152 variants (they differ by a few millimetres) and
    surface-based atlases (Destrieux, Desikan-Killiany) use their volumetric-MNI or
    surface-derived centroids. This is immaterial for visualisation.

---

## Using a custom atlas

Pass a path to a CSV file with **one row per connectome node**. The required and optional
columns are:

| Column | Required | Meaning |
|---|:---:|---|
| `region` | ✅ | Region name (used to label edges and nodes) |
| `x`, `y`, `z` | ✅ | MNI coordinates in millimetres |
| `network` | optional | Functional network label — enables the network-summary matrix and chord diagram |
| `hemisphere` | optional | `L` / `R` (`M` for midline) — used to group/colour nodes |
| `structure` | optional | e.g. `cortical` / `subcortical` / `cerebellum` — used to group nodes |

Example `my_atlas.csv` (a 4-node connectome):

```csv
region,x,y,z,network,hemisphere,structure
L_Visual_1,-14,-100,8,Visual,L,cortical
R_Visual_1,16,-98,6,Visual,R,cortical
L_Motor_1,-38,-22,54,Somatomotor,L,cortical
R_Motor_1,40,-20,52,Somatomotor,R,cortical
```

Then:

```python
cpm = CPMAnalysis(results_directory="results/", atlas="my_atlas.csv")
```

**Row order matters:** row *i* of the atlas describes node *i* of your connectome, in the same
node order used to build `X`. Only `region`, `x`, `y`, `z` are required; add `network` if you
want the network-level figures, and `hemisphere` / `structure` for nicer grouping when the
atlas has no functional-network partition.

If you have coordinates but no region names (or vice versa), you still need all four required
columns — use placeholder names like `Region_1`, `Region_2`, … if necessary.
