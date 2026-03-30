"""Extract SFR training data from Arepo snapshots.

Produces per-snapshot .npz files with:
  context  (n_pixels, 5)       -- 2D galaxy properties per pixel
  sfr      (n_cells_total,)    -- SFR of every Voronoi cell
  pixel_id (n_cells_total,)    -- which pixel each cell belongs to

Context features (columns of the context array):
  0: gas_surfdens    -- gas surface density
  1: star_surfdens   -- stellar surface density
  2: rotcurve        -- circular velocity from 1D rotation curve
  3: kappa           -- epicyclic frequency
  4: dm_voldens      -- dark-matter midplane volume density (KDTree)

Usage:
  python -m sfemulator.data.extract --config configs/etg_vlM.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import yaml
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter as sg
from scipy.spatial import KDTree
from scipy.stats import binned_statistic_2d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Unit conversions (CGS) ────────────────────────────────────────────
KPC_TO_CM = 3.086e21
PC_TO_CM = 3.086e18
MSOL_TO_G = 1.99e33
G_CGS = 6.67e-8
GAMMA = 5.0 / 3.0
KB_CGS = 1.38e-16
MP_CGS = 1.67e-24
MU = 1.4

CONTEXT_NAMES = ["gas_surfdens", "star_surfdens", "rotcurve", "kappa", "dm_voldens"]


# ── Snapshot I/O ──────────────────────────────────────────────────────

def read_particle_data(snap_path: Path, part_type: int) -> Dict[str, np.ndarray] | None:
    """Read coordinates, velocities, masses, and (for gas) SFR from an Arepo snapshot."""
    with h5py.File(snap_path, "r") as f:
        key = f"PartType{part_type}"
        if key not in f:
            return None
        pt = f[key]
        header = f["Header"]
        box_half = 0.5 * header.attrs["BoxSize"]

        to_pos = pt["Coordinates"].attrs["to_cgs"]
        data = {
            "x": (pt["Coordinates"][:, 0] - box_half) * to_pos,
            "y": (pt["Coordinates"][:, 1] - box_half) * to_pos,
            "z": (pt["Coordinates"][:, 2] - box_half) * to_pos,
        }

        to_vel = pt["Velocities"].attrs["to_cgs"]
        data["vx"] = pt["Velocities"][:, 0] * to_vel
        data["vy"] = pt["Velocities"][:, 1] * to_vel
        data["vz"] = pt["Velocities"][:, 2] * to_vel

        if "Masses" in pt:
            data["mass"] = pt["Masses"][:] * pt["Masses"].attrs["to_cgs"]
        else:
            mass_table_val = header.attrs["MassTable"][part_type]
            to_mass = f["PartType0/Masses"].attrs["to_cgs"]
            data["mass"] = np.full(len(data["x"]), mass_table_val * to_mass)

        if part_type == 0:
            data["potential"] = pt["Potential"][:] * pt["Potential"].attrs["to_cgs"]
            data["sfr"] = pt["StarFormationRate"][:] * pt["StarFormationRate"].attrs["to_cgs"]
            data["density"] = pt["Density"][:] * pt["Density"].attrs["to_cgs"]

    data["R"] = np.sqrt(data["x"] ** 2 + data["y"] ** 2)
    return data


# ── Galaxy alignment ──────────────────────────────────────────────────

def compute_gas_com(gas: Dict[str, np.ndarray], xymax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mass-weighted center of mass (position & velocity) of disk gas."""
    cnd = (gas["R"] < xymax) & (np.abs(gas["z"]) < xymax)
    m = gas["mass"][cnd]
    pos_cm = np.array([np.average(gas[k][cnd], weights=m) for k in ("x", "y", "z")])
    vel_cm = np.array([np.average(gas[k][cnd], weights=m) for k in ("vx", "vy", "vz")])
    return pos_cm, vel_cm


def compute_angular_momentum(gas: Dict[str, np.ndarray], pos_cm: np.ndarray, vel_cm: np.ndarray, xymax: float) -> np.ndarray:
    """Total angular momentum vector of disk gas, relative to COM."""
    cnd = (gas["R"] < xymax) & (np.abs(gas["z"]) < xymax)
    m = gas["mass"][cnd]
    r = np.column_stack([gas["x"][cnd] - pos_cm[0], gas["y"][cnd] - pos_cm[1], gas["z"][cnd] - pos_cm[2]])
    v = np.column_stack([gas["vx"][cnd] - vel_cm[0], gas["vy"][cnd] - vel_cm[1], gas["vz"][cnd] - vel_cm[2]])
    L = np.sum(m[:, None] * np.cross(r, v), axis=0)
    return L


def build_rotation_matrix(L: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps L -> z-hat."""
    Lx, Ly, Lz = L
    Lperp = np.sqrt(Lx ** 2 + Ly ** 2)
    zu = L / np.linalg.norm(L)
    xu = np.array([-Ly, Lx, 0.0]) / Lperp
    yu = np.array([-Lx * Lz, -Ly * Lz, Lx ** 2 + Ly ** 2])
    yu = yu / np.linalg.norm(yu)
    return np.array([xu, yu, zu])


def realign_particles(data: Dict[str, np.ndarray], rot: np.ndarray, pos_cm: np.ndarray, vel_cm: np.ndarray) -> None:
    """Shift to COM frame and rotate so disk lies in the x-y plane. Modifies in place."""
    r = np.column_stack([data["x"] - pos_cm[0], data["y"] - pos_cm[1], data["z"] - pos_cm[2]])
    v = np.column_stack([data["vx"] - vel_cm[0], data["vy"] - vel_cm[1], data["vz"] - vel_cm[2]])
    r_rot = r @ rot.T
    v_rot = v @ rot.T
    data["x"], data["y"], data["z"] = r_rot[:, 0], r_rot[:, 1], r_rot[:, 2]
    data["vx"], data["vy"], data["vz"] = v_rot[:, 0], v_rot[:, 1], v_rot[:, 2]
    data["R"] = np.sqrt(data["x"] ** 2 + data["y"] ** 2)


# ── Midplane finding ─────────────────────────────────────────────────

def find_midplane_z(gas: Dict[str, np.ndarray], xbin_edges: np.ndarray, ybin_edges: np.ndarray, total_height_cm: float, zbin_width_cm: float) -> np.ndarray:
    """Find the midplane z-position per (x,y) pixel via the potential minimum."""
    nxy = len(xbin_edges) - 1
    zbin_edges = np.arange(-total_height_cm, total_height_cm + zbin_width_cm, zbin_width_cm)
    nz = len(zbin_edges) - 1
    zbin_centers = (zbin_edges[:-1] + zbin_edges[1:]) / 2.0

    # Assign each gas cell to an (x, y, z) voxel
    ix = np.digitize(gas["x"], xbin_edges) - 1
    iy = np.digitize(gas["y"], ybin_edges) - 1
    iz = np.digitize(gas["z"], zbin_edges) - 1
    valid = (ix >= 0) & (ix < nxy) & (iy >= 0) & (iy < nxy) & (iz >= 0) & (iz < nz)

    # Mass-weighted mean potential per voxel
    pot_sum = np.zeros((nxy, nxy, nz))
    mass_sum = np.zeros((nxy, nxy, nz))
    np.add.at(pot_sum, (ix[valid], iy[valid], iz[valid]), gas["potential"][valid] * gas["mass"][valid])
    np.add.at(mass_sum, (ix[valid], iy[valid], iz[valid]), gas["mass"][valid])

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_pot = np.where(mass_sum > 0, pot_sum / mass_sum, np.inf)

    midplane_iz = np.argmin(mean_pot, axis=2)
    midplane_z = zbin_centers[midplane_iz]

    # Pixels with no gas at all: default to z = 0
    empty = np.all(mass_sum == 0, axis=2)
    midplane_z[empty] = 0.0

    return midplane_z


# ── Context feature computation ──────────────────────────────────────

def compute_surface_density(particles: Dict[str, np.ndarray], xbin_edges: np.ndarray, ybin_edges: np.ndarray, pixel_area: float) -> np.ndarray:
    """Sum of particle masses per pixel, divided by pixel area."""
    surfdens, _, _, _ = binned_statistic_2d(
        particles["x"], particles["y"], particles["mass"],
        bins=(xbin_edges, ybin_edges), statistic="sum",
    )
    return surfdens / pixel_area


def compute_rotation_curve_1d(gas: Dict[str, np.ndarray], rbin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mass-weighted circular velocity in radial bins."""
    rbin_centers = (rbin_edges[:-1] + rbin_edges[1:]) / 2.0
    vcirc = np.full(len(rbin_centers), np.nan)

    for i, (rmin, rmax) in enumerate(zip(rbin_edges[:-1], rbin_edges[1:])):
        cnd = (gas["R"] > rmin) & (gas["R"] < rmax)
        if not np.any(cnd):
            continue
        # v_circ = (-y*vx + x*vy) / R
        vphi = (-gas["y"][cnd] * gas["vx"][cnd] + gas["x"][cnd] * gas["vy"][cnd]) / gas["R"][cnd]
        vcirc[i] = np.average(vphi, weights=gas["mass"][cnd])

    return rbin_centers, vcirc


def interpolate_to_2d(values_1d: np.ndarray, rbin_centers: np.ndarray, x2d: np.ndarray, y2d: np.ndarray) -> np.ndarray:
    """Interpolate a 1D radial profile onto the 2D pixel grid."""
    f = interp1d(rbin_centers, values_1d, bounds_error=False, fill_value=(np.nan, np.nan))
    R_grid = np.sqrt(x2d ** 2 + y2d ** 2)
    return f(R_grid)


def compute_kappa_1d(rbin_centers: np.ndarray, vcirc: np.ndarray, polyno: int = 2, wndwlen: int = 5) -> np.ndarray:
    """Epicyclic frequency from the rotation curve, using Savitzky-Golay smoothing."""
    omega = vcirc / rbin_centers
    dR = sg(rbin_centers, wndwlen, polyno, deriv=1)
    dvc = sg(vcirc, wndwlen, polyno, deriv=1)
    beta = dvc / dR * rbin_centers / vcirc
    kappa = omega * np.sqrt(2.0 * (1.0 + beta))
    return kappa


def compute_dm_voldens(dm: Dict[str, np.ndarray], x2d: np.ndarray, y2d: np.ndarray, midplane_z: np.ndarray, total_height_cm: float, xymax: float, search_radius_cm: float) -> np.ndarray:
    """Dark-matter midplane volume density via KDTree query."""
    cnd = (np.abs(dm["z"]) < total_height_cm + search_radius_cm) & (dm["R"] < xymax + search_radius_cm)
    tree = KDTree(np.column_stack([dm["x"][cnd], dm["y"][cnd], dm["z"][cnd]]))

    grid_points = np.column_stack([x2d.ravel(), y2d.ravel(), midplane_z.ravel()])
    neighbor_lists = tree.query_ball_point(grid_points, search_radius_cm)

    masses_cut = dm["mass"][cnd]
    total_mass = np.array([np.sum(masses_cut[idxs]) for idxs in neighbor_lists])
    voldens = total_mass.reshape(x2d.shape) / search_radius_cm ** 3
    return voldens


# ── Main extractor ────────────────────────────────────────────────────

def merge_particle_dicts(dicts: list[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray] | None:
    """Concatenate particle data from multiple PartTypes into one dict."""
    dicts = [d for d in dicts if d is not None]
    if not dicts:
        return None
    keys = [k for k in dicts[0] if all(k in d for d in dicts)]
    return {k: np.concatenate([d[k] for d in dicts]) for k in keys}


def extract_snapshot(snap_path: Path, xymax_kpc: float, xybin_width_pc: float, total_height_kpc: float, rotcurve_rsln_pc: float, dm_search_radius_kpc: float = 0.5, midplane_zbin_width_pc: float = 10.0, min_nH: float = 0.13) -> Dict[str, np.ndarray]:
    """Extract context features and per-cell SFR from a single snapshot."""
    xymax = xymax_kpc * KPC_TO_CM
    xybin_width = xybin_width_pc * PC_TO_CM
    total_height = total_height_kpc * KPC_TO_CM
    rotcurve_rsln = rotcurve_rsln_pc * PC_TO_CM
    dm_search_radius = dm_search_radius_kpc * KPC_TO_CM
    midplane_zbin_width = midplane_zbin_width_pc * PC_TO_CM
    pixel_area = xybin_width ** 2
    # Convert hydrogen number density threshold to mass density (CGS)
    min_density_cgs = min_nH * MU * MP_CGS

    # Load particles
    # Stars = PartType2 (initial disk) + PartType3 (bulge, if present) + PartType4 (newly formed)
    # PartType5 is Monte Carlo tracers, NOT stars
    logger.info("Loading snapshot: %s", snap_path)
    gas = read_particle_data(snap_path, 0)
    stellar_types = [read_particle_data(snap_path, i) for i in (2, 3, 4)]
    stars = merge_particle_dicts(stellar_types)
    dm = read_particle_data(snap_path, 1)

    # Realign to gas angular momentum
    pos_cm, vel_cm = compute_gas_com(gas, xymax)
    L = compute_angular_momentum(gas, pos_cm, vel_cm, xymax)
    rot = build_rotation_matrix(L)
    realign_particles(gas, rot, pos_cm, vel_cm)
    if stars is not None:
        realign_particles(stars, rot, pos_cm, vel_cm)
    if dm is not None:
        realign_particles(dm, rot, pos_cm, vel_cm)

    # 2D grid
    nxy = int(np.rint(2.0 * xymax / xybin_width))
    xbin_edges = np.linspace(-xymax, xymax, nxy + 1)
    xbin_centers = (xbin_edges[:-1] + xbin_edges[1:]) / 2.0
    ybin_edges = xbin_edges.copy()
    ybin_centers = xbin_centers.copy()
    x2d, y2d = np.meshgrid(xbin_centers, ybin_centers, indexing="ij")

    # Midplane z via potential minimum
    midplane_z = find_midplane_z(gas, xbin_edges, ybin_edges, total_height, midplane_zbin_width)

    # Rotation-curve radial bins (extra leeway for interpolation)
    rmax = xymax + rotcurve_rsln * 5
    n_rbins = int(np.rint(rmax / rotcurve_rsln))
    rbin_edges = np.linspace(0.0, rmax, n_rbins + 1)

    # Density cut: exclude diffuse gas below the threshold from gas surface
    # density and SFR. Rotation curve uses all gas (kinematic quantity).
    dense_gas = {k: v[gas["density"] >= min_density_cgs] for k, v in gas.items()}
    logger.info(
        "Density cut (n_H >= %.2f cm^-3): %d / %d gas cells pass",
        min_nH, len(dense_gas["x"]), len(gas["x"]),
    )

    # Context features
    gas_surfdens = compute_surface_density(dense_gas, xbin_edges, ybin_edges, pixel_area)
    star_surfdens = compute_surface_density(stars, xbin_edges, ybin_edges, pixel_area) if stars is not None else np.zeros_like(gas_surfdens)

    rbin_centers, vcirc = compute_rotation_curve_1d(gas, rbin_edges)
    rotcurve = interpolate_to_2d(vcirc, rbin_centers, x2d, y2d)
    kappa = interpolate_to_2d(compute_kappa_1d(rbin_centers, vcirc), rbin_centers, x2d, y2d)

    dm_voldens = compute_dm_voldens(dm, x2d, y2d, midplane_z, total_height, xymax, dm_search_radius) if dm is not None else np.zeros_like(gas_surfdens)

    # Stack into (n_pixels, 5)
    context = np.column_stack([
        gas_surfdens.ravel(),
        star_surfdens.ravel(),
        rotcurve.ravel(),
        kappa.ravel(),
        dm_voldens.ravel(),
    ])

    # Assign each dense gas cell to a pixel and collect SFR
    ix = np.digitize(dense_gas["x"], xbin_edges) - 1
    iy = np.digitize(dense_gas["y"], ybin_edges) - 1
    in_grid = (ix >= 0) & (ix < nxy) & (iy >= 0) & (iy < nxy)
    pixel_id = ix[in_grid] * nxy + iy[in_grid]
    sfr = dense_gas["sfr"][in_grid]

    logger.info(
        "Extracted: %d pixels (%dx%d), %d cells, %d with SFR>0",
        context.shape[0], nxy, nxy, len(sfr), np.sum(sfr > 0),
    )

    return {"context": context, "sfr": sfr, "pixel_id": pixel_id}


# ── Config & CLI ──────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    """Load a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run(config: dict) -> None:
    """Process all snapshots specified in the config."""
    root_dir = Path(config["root_dir"])
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    subdir = config.get("subdir", "")
    snap_prefix = config.get("snap_prefix", "snap-DESPOTIC")

    snap_numbers = range(config["beg_snap"], config["end_snap"] + 1, config.get("snap_step", 1))

    for snap_no in snap_numbers:
        snap_path = root_dir / subdir / f"{snap_prefix}_{snap_no:03d}.hdf5"
        if not snap_path.exists():
            logger.warning("Snapshot not found, skipping: %s", snap_path)
            continue

        result = extract_snapshot(
            snap_path,
            xymax_kpc=config.get("xymax_kpc", 15.0),
            xybin_width_pc=config.get("xybin_width_pc", 80.0),
            total_height_kpc=config.get("total_height_kpc", 1.5),
            rotcurve_rsln_pc=config.get("rotcurve_rsln_pc", 20.0),
            dm_search_radius_kpc=config.get("dm_search_radius_kpc", 0.5),
            midplane_zbin_width_pc=config.get("midplane_zbin_width_pc", 10.0),
            min_nH=config.get("min_nH", 0.13),
        )

        out_path = save_dir / f"sfr_context_{snap_no:03d}.npz"
        np.savez(out_path, **result)
        logger.info("Saved: %s", out_path)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Extract SFR training data from Arepo snapshots")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
