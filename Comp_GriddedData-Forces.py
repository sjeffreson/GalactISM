import numpy as np
import h5py
from GriddedData import GriddedDataset
import astro_helper as ah

import glob, os, re, sys
from pathlib import Path
import configparser
config = configparser.ConfigParser()
import pickle
import argparse, logging
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''Compute weights, forces and mid-plane indices (according to the potential)
for one galaxy at a time. The mid-plane indices are required to compute the
other quantities for the GalactISM dataset.'''
config.read(sys.argv[1])
galname = sys.argv[2]
params = config[galname]
PROPS = config['GENERAL']['PROPS'].split(', ')
print(PROPS)
PROPSSTRING = config['GENERAL']['PROPSSTRING']

savestring = ""
EXCLUDE_TEMP = params.get('EXCLUDE_TEMP')
if EXCLUDE_TEMP == 'None':
    EXCLUDE_TEMP = None
else:
    EXCLUDE_TEMP = float(EXCLUDE_TEMP)
    savestring += "_T{:.1e}".format(EXCLUDE_TEMP)
EXCLUDE_AVIR = params.get('EXCLUDE_AVIR')
if EXCLUDE_AVIR == 'None':
    EXCLUDE_AVIR = None
else:
    EXCLUDE_AVIR = float(EXCLUDE_AVIR)
    savestring += "_avir{:.1e}".format(EXCLUDE_AVIR)
snapstr = params.get('SNAPSTR')

snapnames = [
    snapstr + "_{0:03d}.hdf5".format(i) for i in 
    range(params.getint('BEGSNAPNO'), params.getint('ENDSNAPNO')+1)
]

props_3D = {key: None for key in PROPS}
for snapname in snapnames:

    '''Load the gal, feed it these mid-plane idcs'''
    gal = GriddedDataset(
        params=params,
        galaxy_type=galname,
        total_height=params.getfloat('TOT_HEIGHT'), # kpc
        zbin_width_ptl=10.,
        xymax=params.getfloat('XYMAX'), # kpc
        snapname=snapname,
        exclude_temp_above=EXCLUDE_TEMP,
        exclude_avir_below=EXCLUDE_AVIR,
        exclude_HII=True,
        realign_galaxy_to_gas=True, # according to angular momentum vector of gas
        required_particle_types=[0,1,2,3,4], # just gas by default
    )

    for prop in PROPS:
        axisnum = 2
        if props_3D[prop] is None:
            props_3D[prop] = gal.get_prop_by_keyword(prop)
        elif props_3D[prop].ndim == axisnum:
            props_3D[prop] = np.stack((props_3D[prop], gal.get_prop_by_keyword(prop)), axis=axisnum)
        else:
            props_3D[prop] = np.concatenate((props_3D[prop], gal.get_prop_by_keyword(prop)[...,np.newaxis]), axis=axisnum)
        logger.info("Adding {0}, shape of {1}: {2}".format(snapname, prop, props_3D[prop].shape))

    '''Save the dictionary to a temporary pickle at regular intervals'''
    filesavedir = Path(params['SAVE_DIR']) / params['SUBDIR']
    if (props_3D[PROPS[-1]].ndim > axisnum) & ((props_3D[PROPS[-1]].shape[-1] % 25 == 0) | (snapname == snapnames[-1])):
        filesavename = str(filesavedir / PROPSSTRING) + "_{:s}_{:s}".format(
            re.search(r'\d+', snapname).group(), galname
        ) + savestring + ".pkl"
        with open(filesavename, "wb") as f:
            pickle.dump(props_3D, f)
        logger.info("Saved: {:s}".format(str(filesavename)))
        for prop in PROPS: # memory
            del props_3D[prop]
            props_3D[prop] = None
    del gal # memory