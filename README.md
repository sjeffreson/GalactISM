# GalactISM

A data extraction pipeline for training normalizing flows on resolved star formation rates in galaxy simulations.

The pipeline takes Arepo hydrodynamic simulation snapshots and produces compact training data: for each 2D pixel on a galaxy's face-on grid, it extracts 5 context features, and for every Voronoi gas cell in that pixel column, it records the cell's star formation rate. A downstream normalizing flow then learns the full conditional distribution P(SFR | context).

## Output format

Each snapshot produces a `.npz` file with three arrays:

| Array | Shape | Description |
|-------|-------|-------------|
| `context` | `(n_pixels, 5)` | 2D context features per pixel |
| `sfr` | `(n_cells,)` | Star formation rate of every Voronoi cell |
| `pixel_id` | `(n_cells,)` | Maps each cell to its pixel in `context` |

To pair cells with their context: `context[pixel_id]` broadcasts the 2D features to every 3D cell.

### Context features

| Column | Name | Description |
|--------|------|-------------|
| 0 | `gas_surfdens` | Gas surface density (sum of cell masses / pixel area) |
| 1 | `star_surfdens` | Stellar surface density |
| 2 | `rotcurve` | Circular velocity from the 1D rotation curve |
| 3 | `kappa` | Epicyclic frequency (Savitzky-Golay smoothed) |
| 4 | `dm_voldens` | Dark-matter midplane volume density (KDTree) |

## Installation

Requires Python 3.10+ and the following packages:

```
numpy
scipy
h5py
pyyaml
```

No package install is needed -- run directly from the repo root.

## Usage

### Extract training data from snapshots

```bash
python -m sfemulator.data.extract --config configs/test.yaml
```

### Configuration

Configs are YAML files. See `configs/` for examples. Key parameters:

```yaml
root_dir: /path/to/simulations    # directory containing snapshot files
save_dir: /path/to/output         # where to write .npz files
subdir: ""                        # subdirectory within root_dir
snap_prefix: snap-DESPOTIC        # snapshot filename prefix

beg_snap: 500                     # first snapshot number
end_snap: 500                     # last snapshot number
snap_step: 1                      # stride

xymax_kpc: 6.0                    # half-width of the 2D grid
xybin_width_pc: 80.0              # pixel size
total_height_kpc: 1.5             # vertical extent for midplane search
rotcurve_rsln_pc: 60.0            # radial bin size for rotation curve
min_nH: 0.13                      # minimum gas density (H atoms / cm^3)
```

### Inspect output

```python
import numpy as np

d = np.load("output/sfr_context_500.npz")
context, sfr, pixel_id = d["context"], d["sfr"], d["pixel_id"]

# Gas surface density for every cell
gas_surfdens_per_cell = context[pixel_id, 0]
```

## Pipeline steps

1. Load snapshot (gas, stars, dark matter) via h5py
2. Realign galaxy to the gas angular momentum vector
3. Find midplane z-position per pixel via gravitational potential minimum
4. Lay down a 2D (x, y) grid and assign gas cells to pixels
5. Apply a density floor (default 0.13 n_H cm^-3) to exclude diffuse background gas
6. Compute the 5 context features per pixel
7. Collect per-cell SFR values with their pixel assignments
8. Save as `.npz`

## Project structure

```
sfemulator/
  data/
    extract.py       # main extraction pipeline
    __main__.py       # CLI entry point
configs/
  test.yaml          # single-snapshot test config
  etg_vlM.yaml       # ETG very-low-mass config
notebooks/
  sfr_vs_surfdens.ipynb  # SFR vs gas surface density plot
```

## Input data

Expects Arepo simulation snapshots in HDF5 format (`snap-DESPOTIC_*.hdf5`) with:
- `PartType0` (gas): Coordinates, Velocities, Masses, Density, Potential, StarFormationRate
- `PartType1` (dark matter): Coordinates, Velocities (masses from MassTable)
- `PartType2` (disk stars): Coordinates, Velocities (masses from MassTable)
- `PartType4` (new stars): Coordinates, Velocities, Masses

## Contributors

- **Sarah Jeffreson** -- project lead, original simulation analysis code
- **Claude** (Anthropic) -- refactored extraction pipeline, README, and notebook

## License

See [LICENSE](LICENSE).
