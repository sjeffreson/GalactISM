"""Load extracted .npz snapshots and prepare training data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

# Unit conversions from CGS
PC_TO_CM = 3.086e18
KPC_TO_CM = 3.086e21
MSOL_TO_G = 1.99e33
YR_TO_S = 3.15576e7
MYR_TO_S = 3.15576e13

SURFDENS_TO_MSUN_PC2 = PC_TO_CM**2 / MSOL_TO_G
SFR_SURFDENS_TO_MSUN_YR_KPC2 = YR_TO_S * KPC_TO_CM**2 / MSOL_TO_G

CONTEXT_NAMES = ["gas_surfdens", "star_surfdens", "rotcurve", "kappa", "dm_voldens"]

GALAXIES = {
    "NGC300":   {"dir": "NGC300",   "pixel_pc": 80.0},
    "ETG-vlM":  {"dir": "ETG-vlM",  "pixel_pc": 80.0},
    "ETG-lowM": {"dir": "ETG-lowM", "pixel_pc": 80.0},
    "ETG-medM": {"dir": "ETG-medM", "pixel_pc": 80.0},
    "ETG-hiM":  {"dir": "ETG-hiM",  "pixel_pc": 80.0},
}


def load_galaxy(name: str, snap_dir: Path, pixel_pc: float = 80.0) -> Tuple[np.ndarray, np.ndarray]:
    """Load all snapshots for one galaxy, return (context, sfr_surfdens) per pixel."""
    info = GALAXIES[name]
    snap_files = sorted((snap_dir / info["dir"]).glob("sfr_context_*.npz"))
    pixel_area = (info["pixel_pc"] * PC_TO_CM) ** 2

    all_context = []
    all_sfr_surfdens = []

    for f in snap_files:
        d = np.load(f)
        context, sfr, pixel_id = d["context"], d["sfr"], d["pixel_id"]
        n_pixels = context.shape[0]

        sfr_sum = np.zeros(n_pixels)
        np.add.at(sfr_sum, pixel_id, sfr)
        sfr_surfdens = sfr_sum / pixel_area

        has_sf = sfr_surfdens > 0
        all_context.append(context[has_sf])
        all_sfr_surfdens.append(sfr_surfdens[has_sf])

    ctx = np.concatenate(all_context)
    sfr_sd = np.concatenate(all_sfr_surfdens)
    print(f"{name}: {len(snap_files)} snapshots, {len(sfr_sd)} star-forming pixels")
    return ctx, sfr_sd


def load_all_galaxies(snap_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load all galaxies from a snapshot directory."""
    return {name: load_galaxy(name, snap_dir) for name in GALAXIES}


def prepare_log_data(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    galaxy_names: list[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine galaxies into log-space arrays, filtering non-finite rows.

    Returns (X_log, y_log, finite_mask_on_original).
    """
    if galaxy_names is None:
        galaxy_names = list(data.keys())
    all_ctx = np.concatenate([data[n][0] for n in galaxy_names])
    all_sfr = np.concatenate([data[n][1] for n in galaxy_names])

    log_ctx = np.log10(all_ctx)
    log_sfr = np.log10(all_sfr)
    finite = np.all(np.isfinite(log_ctx), axis=1) & np.isfinite(log_sfr)

    return log_ctx[finite], log_sfr[finite], finite
