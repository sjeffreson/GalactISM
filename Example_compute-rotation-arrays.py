import glob, os, sys

import h5py
import numpy as np

import astro_helper as ah
from GriddedData import GriddedDataset

DEFAULT_ROOT_DIR = "/n/holystore01/LABS/itc_lab/Users/sjeffreson/"
galaxy_dir = sys.argv[1]
diameter = float(sys.argv[2])
search_str = sys.argv[3]
snapnames = [snapname for snapname in sorted(glob.glob(os.path.join(DEFAULT_ROOT_DIR+galaxy_dir, search_str)))]
snapnos = [snapname.split("_")[-1].split(".")[0] for snapname in snapnames]

for snapname, snapno in zip(snapnames, snapnos):
    data = GriddedDataset(galaxy_dir, max_grid_width=diameter, snapname=snapname, ptl_resolution=10.)

    # galactic angular velocity
    Omegaz_array = data.get_Omegaz_array()
    np.save(DEFAULT_ROOT_DIR + galaxy_dir + '/Omegazs_' + snapno + '.npy', Omegaz_array)
    print("Saved: "+ DEFAULT_ROOT_DIR + galaxy_dir + '/Omegazs_' + snapno + ".npy")

    # epicyclic frequency
    kappa_array = data.get_kappa_array(polyno=2, wndwlen=41)
    np.save(DEFAULT_ROOT_DIR + galaxy_dir + '/kappas_' + snapno + '.npy', kappa_array)
    print("Saved: "+ DEFAULT_ROOT_DIR + galaxy_dir + '/kappas_' + snapno + ".npy")