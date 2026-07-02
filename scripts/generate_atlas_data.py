"""Generate bundled atlas coordinate CSVs from authentic public sources.

This script regenerates the coordinate tables shipped in
``src/cccpm/atlases/data/``. It never invents coordinates: every table is
derived from a published source (CBIG/Yeo for Schaefer, nilearn for the
coordinate-defined atlases). Provenance for each file is written to
``src/cccpm/atlases/data/SOURCES.md``.

Run from the repo root::

    poetry run python scripts/generate_atlas_data.py

Requires network access and ``nilearn``. Files that cannot be reached (e.g. a
source host that is temporarily down) are skipped with a warning; existing
bundled files are left untouched so a partial run never corrupts the bundle.

Output schema (per row = one connectome node), columns:

    region, x, y, z, network, hemisphere, structure

``network`` is omitted for atlases that do not define a functional partition.
``hemisphere`` is ``L``/``R`` (or ``M`` for midline); ``structure`` is
``cortical``/``subcortical``/``cerebellum``. See
``examples/atlas_coordinates_design.md`` for the rationale.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "src" / "cccpm" / "atlases" / "data"

# Collected provenance records: (atlas_name, n_nodes, source_url, citation)
_PROVENANCE: list[tuple[str, int, str, str]] = []


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _hemisphere_from_x(x: float, tol: float = 1.0) -> str:
    """L/R/M from an MNI x coordinate (x < 0 is left)."""
    if x < -tol:
        return "L"
    if x > tol:
        return "R"
    return "M"


def _download(url: str, dest: Path) -> bool:
    """Download ``url`` to ``dest``. Falls back to curl when requests' TLS
    verification fails (some atlas hosts ship an incomplete cert chain that
    curl tolerates but urllib/requests reject)."""
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception:
        try:
            subprocess.run(["curl", "-sL", "--max-time", "180", url, "-o", str(dest)],
                           check=True)
            return dest.exists() and dest.stat().st_size > 0
        except Exception as e:  # noqa: BLE001
            print(f"  download failed for {url}: {e}")
            return False


def _volume_centroids(img):
    """Return (label_values, mm_coords) for a labelled volume: one centre-of-mass
    per non-zero label, converted from voxel to MNI mm via the image affine.

    Centre-of-mass is used instead of ``find_parcellation_cut_coords`` because the
    latter is prohibitively slow for fine parcellations (e.g. Glasser's 360 ROIs)."""
    from scipy import ndimage
    from nilearn import image as nimage
    data = np.asarray(img.dataobj).astype(int)
    labels = [int(v) for v in np.unique(data) if v != 0]
    coms = np.array(ndimage.center_of_mass(np.ones_like(data), data, labels))
    mm = np.array(nimage.coord_transform(coms[:, 0], coms[:, 1], coms[:, 2],
                                         img.affine)).T
    return labels, mm


def _write(name: str, df: pd.DataFrame, source_url: str, citation: str) -> None:
    """Validate and write one atlas table, recording provenance."""
    required = ["region", "x", "y", "z", "hemisphere", "structure"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: generated table missing columns {missing}")
    ordered = [c for c in ["region", "x", "y", "z", "network", "hemisphere", "structure"]
               if c in df.columns]
    df = df[ordered]
    # Round coordinates to 2 dp — sub-mm precision is spurious for centroids.
    for c in ("x", "y", "z"):
        df[c] = df[c].astype(float).round(2)
    out = DATA_DIR / f"{name}.csv"
    df.to_csv(out, index=False)
    _PROVENANCE.append((name, len(df), source_url, citation))
    print(f"  wrote {name}.csv ({len(df)} nodes)")


# ---------------------------------------------------------------------------
# Schaefer (CBIG/Yeo) — flagship, ships functional networks
# ---------------------------------------------------------------------------

# 7- and 17-network short names appear verbatim in the ROI name, so we parse
# the network directly from the label rather than hard-coding a mapping.
_SCHAEFER_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/"
    "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/"
    "Centroid_coordinates/"
    "Schaefer2018_{n}Parcels_{nets}Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
)
_SCHAEFER_CITE = "Schaefer et al. (2018), Cereb Cortex; Yeo et al. (2011), J Neurophysiol."


def generate_schaefer(parcels=(100, 200, 300, 400), networks=(7, 17)) -> None:
    print("Schaefer:")
    for n in parcels:
        for nets in networks:
            url = _SCHAEFER_URL.format(n=n, nets=nets)
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
            except Exception as e:  # noqa: BLE001 — best-effort fetch
                print(f"  SKIP Schaefer{n}-{nets}: {e}")
                continue
            src = pd.read_csv(io.StringIO(resp.text))
            # Columns: ROI Label, ROI Name, R, A, S  (RAS ≈ MNI x/y/z)
            names = src["ROI Name"]
            # e.g. "7Networks_LH_Vis_1" -> hemi=LH, network=Vis
            parts = names.str.split("_", expand=True)
            hemi = parts[1].map({"LH": "L", "RH": "R"}).fillna("M")
            network = parts[2]
            df = pd.DataFrame({
                "region": names,
                "x": src["R"], "y": src["A"], "z": src["S"],
                "network": network,
                "hemisphere": hemi,
                "structure": "cortical",  # Schaefer is a cortical parcellation
            })
            _write(f"Schaefer{n}-{nets}", df, url, _SCHAEFER_CITE)


# ---------------------------------------------------------------------------
# Coordinate-defined atlases via nilearn
# ---------------------------------------------------------------------------

def generate_power264() -> None:
    from nilearn import datasets as nds
    print("Power264:")
    try:
        p = nds.fetch_coords_power_2011()
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP Power264: {e}")
        return
    rois = p["rois"]
    df = pd.DataFrame({
        "region": [f"Power_{int(i)}" for i in rois["roi"]],
        "x": rois["x"], "y": rois["y"], "z": rois["z"],
        "hemisphere": [_hemisphere_from_x(v) for v in rois["x"]],
        "structure": "cortical",
    })
    _write("Power264", df,
           "nilearn.datasets.fetch_coords_power_2011",
           "Power et al. (2011), Neuron.")


def generate_dosenbach160() -> None:
    from nilearn import datasets as nds
    print("Dosenbach160:")
    try:
        d = nds.fetch_coords_dosenbach_2010()
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP Dosenbach160: {e}")
        return
    rois = d["rois"]
    x, y, z = rois["x"].to_numpy(), rois["y"].to_numpy(), rois["z"].to_numpy()
    networks = np.asarray(d["networks"])
    struct = np.where(networks.astype(str) == "cerebellum", "cerebellum", "cortical")
    df = pd.DataFrame({
        "region": np.asarray(d["labels"]),
        "x": x, "y": y, "z": z,
        "network": networks,
        "hemisphere": [_hemisphere_from_x(v) for v in x],
        "structure": struct,
    })
    _write("Dosenbach160", df,
           "nilearn.datasets.fetch_coords_dosenbach_2010",
           "Dosenbach et al. (2010), Science.")


def generate_seitzman300() -> None:
    from nilearn import datasets as nds
    print("Seitzman300:")
    try:
        s = nds.fetch_coords_seitzman_2018()
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP Seitzman300: {e}")
        return
    rois = s["rois"]
    x, y, z = rois["x"].to_numpy(), rois["y"].to_numpy(), rois["z"].to_numpy()
    # `regions` is e.g. cortexL / cortexR / subcortical / cerebellum
    regions = np.asarray(s["regions"])
    struct = np.where(np.char.startswith(regions.astype(str), "cortex"), "cortical",
                      np.where(regions.astype(str) == "cerebellum", "cerebellum",
                               "subcortical"))
    hemi = []
    for reg, xv in zip(regions.astype(str), x):
        if reg.endswith("L"):
            hemi.append("L")
        elif reg.endswith("R"):
            hemi.append("R")
        else:
            hemi.append(_hemisphere_from_x(xv))
    df = pd.DataFrame({
        "region": [f"Seitzman_{i}" for i in range(len(x))],
        "x": x, "y": y, "z": z,
        "network": np.asarray(s["networks"]),
        "hemisphere": hemi,
        "structure": struct,
    })
    _write("Seitzman300", df,
           "nilearn.datasets.fetch_coords_seitzman_2018",
           "Seitzman et al. (2018), NeuroImage.")


# ---------------------------------------------------------------------------
# Volumetric parcellations via nilearn -> MNI centroids
#
# These are atlases whose "native" form is often a FreeSurfer surface
# annotation (Destrieux == aparc.a2009s) or a bilateral volume (Harvard-Oxford).
# We use their published *volumetric MNI* versions and compute one centroid per
# parcel with nilearn — the reliable way to get MNI coordinates for such atlases.
# ---------------------------------------------------------------------------

def generate_destrieux148() -> None:
    from nilearn import datasets as nds
    from nilearn.plotting import find_parcellation_cut_coords
    print("Destrieux148:")
    try:
        d = nds.fetch_atlas_destrieux_2009(legacy_format=False)
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP Destrieux148: {e}")
        return
    coords, labels_out = find_parcellation_cut_coords(d["maps"], return_label_names=True)
    names = d["labels"]["name"]  # DataFrame indexed by integer label
    region = [str(names.loc[int(v)]) for v in labels_out]
    # Names are prefixed "L "/"R " for the two hemispheres.
    hemi = ["L" if r.startswith("L ") else "R" if r.startswith("R ") else "M"
            for r in region]
    df = pd.DataFrame({
        "region": region,
        "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2],
        "hemisphere": hemi,
        "structure": "cortical",
    })
    _write("Destrieux148", df,
           "nilearn.datasets.fetch_atlas_destrieux_2009 (volumetric MNI) "
           "+ find_parcellation_cut_coords",
           "Destrieux et al. (2010), NeuroImage (aparc.a2009s).")


def generate_harvard_oxford() -> None:
    from nilearn import datasets as nds
    from nilearn.plotting import find_parcellation_cut_coords
    print("HarvardOxfordCortical:")
    try:
        # symmetric_split gives separate left/right parcels (one node per side).
        ho = nds.fetch_atlas_harvard_oxford(
            "cort-maxprob-thr25-2mm", symmetric_split=True)
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP HarvardOxfordCortical: {e}")
        return
    coords, labels_out = find_parcellation_cut_coords(ho["maps"], return_label_names=True)
    labels = list(ho["labels"])  # index aligns with integer label value
    region = [str(labels[int(v)]) for v in labels_out]
    hemi = ["L" if r.startswith("Left") else "R" if r.startswith("Right") else "M"
            for r in region]
    df = pd.DataFrame({
        "region": region,
        "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2],
        "hemisphere": hemi,
        "structure": "cortical",
    })
    _write("HarvardOxfordCortical", df,
           "nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', "
           "symmetric_split=True) + find_parcellation_cut_coords",
           "Desikan et al. (2006) / Makris et al. (2006), Harvard-Oxford atlas.")


# ---------------------------------------------------------------------------
# AAL and Glasser — volumetric MNI atlases with per-region XML label names.
# Fetched directly (nilearn's AAL fetch fails on the host's broken TLS chain;
# the Glasser MMP is not distributed via nilearn/templateflow as the 360-parcel
# volume), then centroids are computed from the label volume.
# ---------------------------------------------------------------------------

def _xml_index_to_name(xml_path: Path) -> dict[int, str]:
    """Parse an FSL-style atlas XML into {integer label index -> region name}."""
    root = ET.parse(xml_path).getroot()
    out: dict[int, str] = {}
    for lab in root.iter("label"):
        idx = lab.get("index")
        if idx is None:
            # AAL-style: <label><index>N</index><name>..</name></label>
            i, n = lab.find("index"), lab.find("name")
            if i is not None and n is not None:
                out[int(i.text)] = n.text
        elif lab.text:
            # Glasser-style: <label index="N" ...>Name</label>
            out[int(idx)] = lab.text
    return out


def _aal_structure(name: str) -> str:
    n = name.lower()
    if n.startswith("cerebel") or n.startswith("vermis"):
        return "cerebellum"
    subcortical = ("caudate", "putamen", "pallidum", "thalamus",
                   "amygdala", "hippocampus")
    if any(k in n for k in subcortical):
        return "subcortical"
    return "cortical"


def generate_aal116() -> None:
    from nilearn import image
    print("AAL116:")
    url = "https://www.gin.cnrs.fr/AAL_files/aal_for_SPM12.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tarball = tmp / "aal.tar.gz"
        if not _download(url, tarball):
            print("  SKIP AAL116: download failed")
            return
        with tarfile.open(tarball) as tf:
            tf.extractall(tmp)  # noqa: S202 — trusted, well-known archive
        nii = tmp / "aal" / "atlas" / "AAL.nii"
        xml = tmp / "aal" / "atlas" / "AAL.xml"
        if not nii.exists():
            print("  SKIP AAL116: AAL.nii not found in archive")
            return
        labels, mm = _volume_centroids(image.load_img(str(nii)))
        idx2name = _xml_index_to_name(xml)
    region = [idx2name.get(v, f"AAL_{v}") for v in labels]
    hemi = ["L" if r.endswith("_L") else "R" if r.endswith("_R") else "M"
            for r in region]
    df = pd.DataFrame({
        "region": region,
        "x": mm[:, 0], "y": mm[:, 1], "z": mm[:, 2],
        "hemisphere": hemi,
        "structure": [_aal_structure(r) for r in region],
    })
    _write("AAL116", df, url,
           "Tzourio-Mazoyer et al. (2002), NeuroImage; AAL for SPM12.")


def generate_glasser360() -> None:
    from nilearn import image
    print("Glasser360:")
    base = ("https://raw.githubusercontent.com/mbedini/"
            "The-HCP-MMP1.0-atlas-in-FSL/master")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nii = tmp / "glasser.nii.gz"
        xml = tmp / "glasser.xml"
        ok = _download(f"{base}/MNI_Glasser_HCP_v1.0.nii.gz", nii)
        ok &= _download(f"{base}/HCP-Multi-Modal-Parcellation-1.0.xml", xml)
        if not ok:
            print("  SKIP Glasser360: download failed")
            return
        labels, mm = _volume_centroids(image.load_img(str(nii)))
        idx2name = _xml_index_to_name(xml)
    region = [idx2name.get(v, f"Glasser_{v}") for v in labels]
    # Names are L_*/R_* (e.g. L_V1, R_p24).
    hemi = ["L" if r.startswith("L_") else "R" if r.startswith("R_") else "M"
            for r in region]
    df = pd.DataFrame({
        "region": region,
        "x": mm[:, 0], "y": mm[:, 1], "z": mm[:, 2],
        "hemisphere": hemi,
        "structure": "cortical",
    })
    _write("Glasser360", df,
           f"{base}/MNI_Glasser_HCP_v1.0.nii.gz",
           "Glasser et al. (2016), Nature; volumetric MNI version "
           "Bedini et al. (2023), Brain Struct Funct.")


# ---------------------------------------------------------------------------
# Provenance sidecar
# ---------------------------------------------------------------------------

def write_sources() -> None:
    lines = [
        "# Bundled atlas coordinate sources",
        "",
        f"Generated by `scripts/generate_atlas_data.py` on {date.today().isoformat()}.",
        "Coordinates are in MNI space (mm). This file is auto-generated — do not edit by hand.",
        "",
        "| Atlas | Nodes | Source | Citation |",
        "|---|---|---|---|",
    ]
    for name, n, url, cite in sorted(_PROVENANCE):
        lines.append(f"| `{name}` | {n} | {url} | {cite} |")
    lines.append("")
    (DATA_DIR / "SOURCES.md").write_text("\n".join(lines))
    print(f"wrote SOURCES.md ({len(_PROVENANCE)} atlases)")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    generate_schaefer()
    generate_power264()
    generate_dosenbach160()
    generate_seitzman300()
    generate_destrieux148()
    generate_harvard_oxford()
    generate_aal116()
    generate_glasser360()
    if not _PROVENANCE:
        print("No atlases generated (network unavailable?).", file=sys.stderr)
        return 1
    write_sources()
    print(f"\nDone: {len(_PROVENANCE)} atlas files in {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
