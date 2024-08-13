from typing import Dict, List, Tuple
from pathlib import Path
import sys
import numpy as np

from scipy.signal import savgol_filter as sg
from scipy.stats import binned_statistic_2d
from scipy.interpolate import interp1d
from scipy.interpolate import RBFInterpolator

import h5py
import astro_helper as ah

import configparser
import argparse, logging
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_ROOT_DIR = Path("/n/holystore01/LABS/itc_lab/Users/sjeffreson/")
class GriddedDataset:
    '''Grid Voronoi cells into a 3D array of desired resolution and calculate
    relevant training data for the normalizing flows, at this resolution.'''
    def __init__(
        self,
        params: configparser.SectionProxy,
        galaxy_type: str,
        total_height: float = 1., # kpc, careful choosing this, we want a z-bin centered on z=0, 25 bins of 80pc
        xymax: float = 15., # kpc
        xybin_width: float = 80., # pc
        zbin_width: float = 80., # pc
        zbin_width_ptl: float = 10., # pc, for computation of the potential for weights only (need finer rsln to capture dens. fluct.)
        rotcurve_rsln: float = 20., # bin resolution for computation of the rotation curve
        exclude_temp_above: float = None, # K
        exclude_avir_below: float = None, # virial parameter
        exclude_HII: bool = False, # whether to completely exclude ionized gas
        snapname: str = "snap-DESPOTIC_300.hdf5",
        midplane_idcs: np.array = None, # if mid-plane already known
        realign_galaxy_to_gas: bool=True, # according to angular momentum vector of gas
        realign_galaxy_to_disk: bool=False, # according to angular momentum vector of entire disk system
        required_particle_types: List[int] = [0], # just gas by default
    ):
        self.ROOT_DIR = Path(params['ROOT_DIR'])
        self.galaxy_type = galaxy_type
        self.galaxy_dir = params['SUBDIR']

        self.total_height = total_height * ah.kpc_to_cm
        self.xymax = xymax * ah.kpc_to_cm
        self.xybin_width = xybin_width * ah.pc_to_cm
        self.zbin_width = zbin_width * ah.pc_to_cm
        self.zbin_width_ptl = zbin_width_ptl * ah.pc_to_cm
        self.rotcurve_rsln = rotcurve_rsln * ah.pc_to_cm
        self.exclude_temp_above = exclude_temp_above
        self.exclude_avir_below = exclude_avir_below
        self.exclude_HII = exclude_HII
        self.snapname = snapname
        self.midplane_idcs = midplane_idcs
        self.realign_galaxy_to_gas = realign_galaxy_to_gas
        self.realign_galaxy_to_disk = realign_galaxy_to_disk
        self._weight_integrand = None

        '''Load all required data'''
        self.data = {}
        for part_type in required_particle_types:
            self.data[part_type] = self._read_snap_data(part_type)

        '''Sixth PartType for all stars'''
        present_stellar_types = [key for key, value in self.data.items() if key in [2, 3, 4] and value is not None]
        if len(present_stellar_types) > 0:
            self.data[5] = {key: np.concatenate([self.data[i][key] for i in present_stellar_types]) for key in self.data[present_stellar_types[0]]}

        '''Seventh PartType for gas that's cut to parameter thresholds, create new variable
        for this, as we don't want to cut out these gas cells for every method.'''
        self.data[6] = None
        cnd = np.ones(len(self.data[0]["R_coords"]), dtype=bool)
        if exclude_temp_above is not None:
            cnd = cnd & (self.data[0]["temps"] < exclude_temp_above)
        if exclude_avir_below is not None:
            cnd = cnd & ~((self.data[0]["AlphaVir"] < exclude_avir_below) & (self.data[0]["AlphaVir"] > 0.))
        self.data[6] = {key: value[cnd] for key, value in self.data[0].items()}
        if exclude_HII:
            self.data[6]["masses"] = self.data[6]["masses"] * (1. - self.data[6]["xHP"])
            self.data[6]["Density"] = self.data[6]["Density"] * (1. - self.data[6]["xHP"])

        '''Realign the galaxy according to the gas or gas+stellar disk'''
        if self.realign_galaxy_to_gas & self.realign_galaxy_to_disk:
            raise ValueError("Galaxy cannot be realigned to both gas and disk. Please choose one.")
        if self.realign_galaxy_to_gas:
            if 0 not in self.data:
                raise ValueError("Gas data must be loaded to realign the galaxy to the gas.")
            self.x_CM, self.y_CM, self.z_CM, self.vx_CM, self.vy_CM, self.vz_CM = self._get_gas_disk_COM()
            self.Lx, self.Ly, self.Lz = self._get_gas_disk_angmom()
            self.data = {key: self._set_realign_galaxy(value) for key, value in self.data.items()}
        elif self.realign_galaxy_to_disk:
            if 0 not in self.data or 2 not in self.data:
                raise ValueError("Gas and stellar disk particles must be loaded to realign the galaxy to the disk.")
            self.x_CM, self.y_CM, self.z_CM, self.vx_CM, self.vy_CM, self.vz_CM = self._get_disk_COM()
            self.Lx, self.Ly, self.Lz = self._get_disk_angmom()
            self.data = {key: self._set_realign_galaxy(value) for key, value in self.data.items()}

        '''Grid on which to compute arrays'''
        self.xybinno = int(np.rint(2.*self.xymax/self.xybin_width))
        self.xbin_edges = np.linspace(-self.xymax, self.xymax, self.xybinno+1)
        self.xbin_centers = (self.xbin_edges[1:]+self.xbin_edges[:-1])/2.

        self.ybin_edges = self.xbin_edges.copy()
        self.ybin_centers = self.xbin_centers.copy()

        self.zbinno = int(np.rint(2.*self.total_height/self.zbin_width))
        self.zbin_centers = np.linspace(-self.total_height, self.total_height, self.zbinno+1)
        self.zbin_edges = (self.zbin_centers[1:]+self.zbin_centers[:-1])/2.

        self.xbin_centers_2d, self.ybin_centers_2d = np.meshgrid(self.xbin_centers, self.ybin_centers)
        self.xbin_centers_2d = self.xbin_centers_2d.T
        self.ybin_centers_2d = self.ybin_centers_2d.T

        '''R-bins for the rotation curve'''
        self.xymax = self.xymax
        self.Rbinno = int(np.rint(self.xymax/self.rotcurve_rsln))
        self.Rbin_edges = np.linspace(0., self.xymax, self.Rbinno+1)
        self.Rbin_centers = (self.Rbin_edges[1:]+self.Rbin_edges[:-1])/2.

        '''Finer z-grid for the potential'''
        self.zbinno_ptl = int(np.rint(2.*self.total_height/self.zbin_width_ptl))
        self.zbin_edges_ptl = np.linspace(-self.total_height, self.total_height, self.zbinno_ptl+1)
        self.zbin_centers_ptl = (self.zbin_edges_ptl[1:]+self.zbin_edges_ptl[:-1])/2.

        self.xbin_centers_3d_ptl, self.ybin_centers_3d_ptl, self.zbin_centers_3d_ptl = np.meshgrid(self.xbin_centers, self.ybin_centers, self.zbin_centers_ptl)
        self.xbin_centers_3d_ptl = np.transpose(self.xbin_centers_3d_ptl, axes=(1, 0, 2))
        self.ybin_centers_3d_ptl = np.transpose(self.ybin_centers_3d_ptl, axes=(1, 0, 2))
        self.zbin_centers_3d_ptl = np.transpose(self.zbin_centers_3d_ptl, axes=(1, 0, 2))
    
        '''Keywords to access methods'''
        self._method_map = {
            'PtlMinIdcs': lambda: self.get_int_force_left_right_xy(PartType=6)[2],
            'weight': lambda: self.get_weight_xy(PartType=6) / ah.kB_cgs,
            'ForceLeft': lambda: self.get_int_force_left_right_xy(PartType=6)[0] / ah.kB_cgs, # alternative way to compute weight
            'ForceRight': lambda: self.get_int_force_left_right_xy(PartType=6)[1] / ah.kB_cgs,
            'Force': lambda: self.get_force_xy(PartType=6) / ah.kB_cgs,
            'SFR_surfdens': lambda: self.get_SFR_surfdens_xy(PartType=6),
            'H2_frac': lambda: self.get_H2_mass_frac_xy(PartType=6),
            'HI_frac': lambda: self.get_HI_mass_frac_xy(PartType=6),
            'gas_surfdens': lambda: self.get_surfdens_xy(PartType=6),
            'star_surfdens': lambda: self.get_surfdens_xy(PartType=5),
            'rotcurve': lambda: self.get_rotation_curve_xy(),
            'kappa': lambda: self.get_kappa_xy(),
            'gas_voldens_midplane': lambda: self.get_midplane_density_xy(PartType=6),
            'star_voldens_midplane': lambda: self.get_midplane_density_xy(PartType=5),
            'veldisp_midplane': lambda: self.get_gas_midplane_veldisps_xyz_xy(PartType=6)[2],
            'Pturb': lambda: self.get_gas_midplane_turbpress_xy(PartType=6),
            'Ptherm': lambda: self.get_gas_midplane_thermpress_xy(PartType=6),
            'SFR_voldens_3D': lambda: self.get_SFR_voldens_xyz(PartType=6), # for later
            # 'H2_frac_3D': lambda: self.get_H2_mass_frac_xyz(),
            # 'HI_frac_3D': lambda: self.get_HI_mass_frac_xyz(),
            # 'Phi': lambda:self.get_potential_xyz(),
            'gas_SF_voldens': lambda: self.get_density_xyz(PartType=6, SFR_cnd=True),
            # 'star_voldens': lambda: self.get_density_xyz(PartType=5),
        }

    def get_prop_by_keyword(self, keyword: str) -> np.array:
        '''Get the physical property array by keyword'''

        if keyword not in self._method_map:
            raise ValueError("Keyword {:s} not found in method map.".format(keyword))
        return self._method_map[keyword]()

    def get_grid(self) -> Tuple[np.array, np.array, np.array]:
        return self.xbin_centers, self.ybin_centers, self.zbin_centers

    ###----------- Data manipulation functions -----------###

    def _read_snap_data(
        self,
        PartType: int,
    ) -> Dict[str, np.array]:
        """Read necessary information about a given particle type, from Arepo snapshot
        Args:
            snapshot (int): snapshot number
            PartType (int): particle type, as in Arepo snapshot
        Returns:
            Dict: dictionary with only the relevant gas information, in cgs units
        """
        snapshot = h5py.File(self.ROOT_DIR / self.galaxy_dir / self.snapname, "r")
        header = snapshot["Header"]
        if "PartType"+str(PartType) not in snapshot:
            return None
        else:
            PartType_data = snapshot["PartType"+str(PartType)]

        snap_data = {}
        snap_data["x_coords"] = (PartType_data['Coordinates'][:,0] - 0.5 * header.attrs['BoxSize']) * PartType_data['Coordinates'].attrs['to_cgs']
        snap_data["y_coords"] = (PartType_data['Coordinates'][:,1] - 0.5 * header.attrs['BoxSize']) * PartType_data['Coordinates'].attrs['to_cgs']
        snap_data["R_coords"] = np.sqrt(snap_data["x_coords"]**2 + snap_data["y_coords"]**2)
        snap_data["phi_coords"] = np.arctan2(snap_data["y_coords"], snap_data["x_coords"])
        snap_data["z_coords"] = (PartType_data['Coordinates'][:,2] - 0.5 * header.attrs['BoxSize']) * PartType_data['Coordinates'].attrs['to_cgs']
        snap_data["velxs"] = PartType_data['Velocities'][:,0] * PartType_data['Velocities'].attrs['to_cgs']
        snap_data["velys"] = PartType_data['Velocities'][:,1] * PartType_data['Velocities'].attrs['to_cgs']
        snap_data["velzs"] = PartType_data['Velocities'][:,2] * PartType_data['Velocities'].attrs['to_cgs']
        snap_data["Potential"] = PartType_data['Potential'][:] * PartType_data['Potential'].attrs['to_cgs']
        if 'Masses' in PartType_data:
            snap_data["masses"] = PartType_data['Masses'][:] * PartType_data['Masses'].attrs['to_cgs']
        else:
            snap_data["masses"] = np.ones(len(snap_data["x_coords"])) * header.attrs['MassTable'][PartType] * snapshot['PartType0/Masses'].attrs['to_cgs']
        if PartType != 0:
            snapshot.close()
            return snap_data
        else:
            # standard Arepo
            snap_data["U"] = PartType_data['InternalEnergy'][:] * PartType_data['InternalEnergy'].attrs['to_cgs']
            snap_data["temps"] = (ah.gamma - 1.) * snap_data["U"] / ah.kB_cgs * ah.mu * ah.mp_cgs
            snap_data["Density"] = PartType_data['Density'][:] * PartType_data['Density'].attrs['to_cgs']
            snap_data["SFRs"] = PartType_data['StarFormationRate'][:] * PartType_data['StarFormationRate'].attrs['to_cgs']
            # specific to Jeffreson et al. runs
            try:
                snap_data["xH2"] = PartType_data['ChemicalAbundances'][:,0] * 2.
                snap_data["xHP"] = PartType_data['ChemicalAbundances'][:,1]
                snap_data["xHI"] = 1. - snap_data["xH2"] - snap_data["xHP"]
                snap_data["AlphaVir"] = PartType_data['AlphaVir'][:]
            except KeyError:
                snap_data["xH2"] = np.zeros(len(snap_data["x_coords"]))
                snap_data["xHP"] = np.zeros(len(snap_data["x_coords"]))
                snap_data["xHI"] = np.ones(len(snap_data["x_coords"]))
                snap_data["AlphaVir"] = np.zeros(len(snap_data["x_coords"]))
            snapshot.close()
            return snap_data
    
    def _cut_out_particles(self, PartType: int=0) -> Dict[str, np.array]:
        '''Cut out most of the gas cells that are in the background grid, not the disk'''

        cnd = (self.data[PartType]["R_coords"] < self.xymax) & (np.fabs(self.data[PartType]["z_coords"]) < self.xymax)
        return {key: value[cnd] for key, value in self.data[PartType].items()}

    def _get_disk_COM(self) -> Tuple[float, float, float, float, float, float]:
        '''Get the center of mass (positional and velocity) of the gas/stellar disk.'''

        gasstar_data = {key: np.concatenate([self.data[2][key], self._cut_out_particles(PartType=0)[key]]) for key in self.data[2]}

        x_CM = np.average(gasstar_data["x_coords"], weights=gasstar_data["masses"])
        y_CM = np.average(gasstar_data["y_coords"], weights=gasstar_data["masses"])
        z_CM = np.average(gasstar_data["z_coords"], weights=gasstar_data["masses"])
        vx_CM = np.average(gasstar_data["velxs"], weights=gasstar_data["masses"])
        vy_CM = np.average(gasstar_data["velys"], weights=gasstar_data["masses"])
        vz_CM = np.average(gasstar_data["velzs"], weights=gasstar_data["masses"])

        return x_CM, y_CM, z_CM, vx_CM, vy_CM, vz_CM

    def _get_gas_disk_COM(self) -> Tuple[float, float, float, float, float, float]:
        '''Get the center of mass (positional and velocity) of the gas disk'''

        gas_data_cut = self._cut_out_particles(PartType=0)

        x_CM = np.average(gas_data_cut["x_coords"], weights=gas_data_cut["masses"])
        y_CM = np.average(gas_data_cut["y_coords"], weights=gas_data_cut["masses"])
        z_CM = np.average(gas_data_cut["z_coords"], weights=gas_data_cut["masses"])
        vx_CM = np.average(gas_data_cut["velxs"], weights=gas_data_cut["masses"])
        vy_CM = np.average(gas_data_cut["velys"], weights=gas_data_cut["masses"])
        vz_CM = np.average(gas_data_cut["velzs"], weights=gas_data_cut["masses"])

        return x_CM, y_CM, z_CM, vx_CM, vy_CM, vz_CM

    def _get_disk_angmom(self) -> Tuple[float, float, float]:
        '''Get the angular momentum vector of the disk'''

        x_CM, y_CM, z_CM, vx_CM, vy_CM, vz_CM = self._get_disk_COM()
        gasstar_data = {key: np.concatenate([self.data[2][key], self._cut_out_particles(PartType=0)[key]]) for key in self.data[2]}

        Lx = np.sum(
            gasstar_data["masses"]*((gasstar_data["y_coords"]-y_CM)*(gasstar_data["velzs"]-vz_CM) -
            (gasstar_data["z_coords"]-z_CM)*(gasstar_data["velys"]-vy_CM))
        )
        Ly = np.sum(
            gasstar_data["masses"]*((gasstar_data["z_coords"]-z_CM)*(gasstar_data["velxs"]-vx_CM) -
            (gasstar_data["x_coords"]-x_CM)*(gasstar_data["velzs"]-vz_CM))
        )
        Lz = np.sum(
            gasstar_data["masses"]*((gasstar_data["x_coords"]-x_CM)*(gasstar_data["velys"]-vy_CM) -
            (gasstar_data["y_coords"]-y_CM)*(gasstar_data["velxs"]-vx_CM))
        )
        return Lx, Ly, Lz

    def _get_gas_disk_angmom(self) -> Tuple[float, float, float]:
        '''Get the angular momentum vector of the gas disk'''

        x_CM, y_CM, z_CM, vx_CM, vy_CM, vz_CM = self._get_gas_disk_COM()
        gas_data_cut = self._cut_out_particles(PartType=0)

        Lx = np.sum(
            gas_data_cut["masses"]*((gas_data_cut["y_coords"]-y_CM)*(gas_data_cut["velzs"]-vz_CM) -
            (gas_data_cut["z_coords"]-z_CM)*(gas_data_cut["velys"]-vy_CM))
        )
        Ly = np.sum(
            gas_data_cut["masses"]*((gas_data_cut["z_coords"]-z_CM)*(gas_data_cut["velxs"]-vx_CM) -
            (gas_data_cut["x_coords"]-x_CM)*(gas_data_cut["velzs"]-vz_CM))
        )
        Lz = np.sum(
            gas_data_cut["masses"]*((gas_data_cut["x_coords"]-x_CM)*(gas_data_cut["velys"]-vy_CM) -
            (gas_data_cut["y_coords"]-y_CM)*(gas_data_cut["velxs"]-vx_CM))
        )
        return Lx, Ly, Lz

    def _set_realign_galaxy(self, snap_data: Dict[str, np.array]) -> Dict[str, np.array]:
        '''Realign the galaxy according to the center of mass and the angular momentum
        vector of the gas disk'''

        if snap_data is None:
            return None

        # new unit vectors
        zu = np.array([self.Lx, self.Ly, self.Lz])/np.sqrt(self.Lx**2+self.Ly**2+self.Lz**2)
        xu = np.array([-self.Ly, self.Lx, 0.]/np.sqrt(self.Lx**2+self.Ly**2))
        yu = np.array([-self.Lx*self.Lz, -self.Ly*self.Lz, self.Lx**2+self.Ly**2])/np.sqrt((-self.Lx*self.Lz)**2+(-self.Ly*self.Lz)**2+(self.Lx**2+self.Ly**2)**2)

        # new co-ordinates
        x = snap_data["x_coords"] - self.x_CM
        y = snap_data["y_coords"] - self.y_CM
        z = snap_data["z_coords"] - self.z_CM
        vx = snap_data["velxs"] - self.vx_CM
        vy = snap_data["velys"] - self.vy_CM
        vz = snap_data["velzs"] - self.vz_CM
        snap_data['x_coords'] = xu[0]*x + xu[1]*y + xu[2]*z
        snap_data['y_coords'] = yu[0]*x + yu[1]*y + yu[2]*z
        snap_data['R_coords'] = np.sqrt(snap_data['x_coords']**2 + snap_data['y_coords']**2)
        snap_data['z_coords'] = zu[0]*x + zu[1]*y + zu[2]*z
        snap_data['velxs'] = xu[0]*vx + xu[1]*vy + xu[2]*vz
        snap_data['velys'] = yu[0]*vx + yu[1]*vy + yu[2]*vz
        snap_data['velzs'] = zu[0]*vx + zu[1]*vy + zu[2]*vz

        snap_data['R_coords'] = np.sqrt(snap_data['x_coords']**2 + snap_data['y_coords']**2)
        snap_data['phi_coords'] = np.arctan2(snap_data['y_coords'], snap_data['x_coords'])

        return snap_data

    def get_data(self) -> Dict[int, Dict[str, np.array]]:
        return self.data

    def get_grid(self) -> Tuple[np.array, np.array, np.array]:
        return self.xbin_centers, self.ybin_centers, self.zbin_centers

    def get_keys(self):
            for key in vars(self).keys():
                print(key)

    def _select_predef_midplane(self, input_array: np.array) -> np.array:
        '''If the mid-plane of the galaxy is already known from a previous calculation (e.g. from the minimum
        of the gravitational potential), select the values at the mid-plane of the array.'''

        midplane_value = np.zeros_like(self.midplane_idcs) * np.nan
        for i in range(self.xybinno):
            for j in range(self.xybinno):
                midplane_value[i,j] = input_array[i,j,self.midplane_idcs[i,j]]

        return midplane_value

    ###----------- LR reliable features -----------###

    def get_surfdens_xy(self, PartType: int=None) -> np.array:
        '''Get the 2D surface density in cgs'''
        if PartType==None:
            logger.critical("Please specify a particle type for get_surfdens_xy.")
        surfdens, _, _, _ = binned_statistic_2d(
            self.data[PartType]["x_coords"], self.data[PartType]["y_coords"], self.data[PartType]["masses"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return surfdens / (self.xybin_width * self.xybin_width)

    def _get_rotation_curve_R(self) -> np.array:
        '''Get the 1D rotation curve of the galaxy, within the gas disk, in cm/s.'''

        vcs = []
        for Rbmin, Rbmax in zip(self.Rbin_edges[:-1], self.Rbin_edges[1:]):
            cnd = (self.data[0]['R_coords'] > Rbmin) & (self.data[0]['R_coords'] < Rbmax)
            if(len(self.data[0]['R_coords'][cnd])>0):
                vcs.append(np.average(
                    -self.data[0]['y_coords'][cnd]/self.data[0]['R_coords'][cnd] * self.data[0]['velxs'][cnd] +
                    self.data[0]['x_coords'][cnd]/self.data[0]['R_coords'][cnd] * self.data[0]['velys'][cnd],
                    weights=self.data[0]['masses'][cnd]))
            else:
                vcs.append(np.nan)

        return np.array(vcs)

    def _get_Omegaz_R(self) -> np.array:
        '''Get the 1D galactic angular velocity in the z-direction, in /s.'''

        vcs = self._get_rotation_curve_R()
        Omegazs = vcs / self.Rbin_centers

        return Omegazs
    
    def _get_kappa_R(self, polyno: int=2, wndwlen: int=5) -> np.array:
        '''Get the 1D epicyclic frequency in the z-direction, in /s.'''

        vcs = self._get_rotation_curve_R()
        Omegazs = vcs / self.Rbin_centers

        dR = sg(self.Rbin_centers, wndwlen, polyno, deriv=1)
        dvc = sg(vcs, wndwlen, polyno, deriv=1)
        betas = dvc/dR * self.Rbin_centers/vcs
        kappas = Omegazs * np.sqrt(2.*(1.+betas))

        return kappas

    def get_rotation_curve_xy(self) -> np.array:
        '''Get the rotation curve of the galaxy by interpolating the 1D rotation curve, in cm/s'''
        vc_R = self._get_rotation_curve_R()

        fvc = interp1d(self.Rbin_centers, vc_R, bounds_error=False, fill_value=(np.nan, np.nan))
        R_grid = np.sqrt(self.xbin_centers_2d**2 + self.ybin_centers_2d**2)
        return fvc(R_grid)

    def get_Omegaz_xy(self) -> np.array:
        '''Get the galactic angular velocity of the galaxy by interpolating the 1D rotation curve, in cm/s'''
        Omegaz_R = self._get_Omegaz_R()

        fOmegaz = interp1d(self.Rbin_centers, Omegaz_R, bounds_error=False, fill_value=(np.nan, np.nan))
        R_grid = np.sqrt(self.xbin_centers_2d**2 + self.ybin_centers_2d**2)
        return fOmegaz(R_grid)
    
    def get_kappa_xy(self, polyno: int=2, wndwlen: int=5) -> np.array:
        '''Get the epicyclic frequency of the galaxy by interpolating the 1D rotation curve, in cm/s'''
        kappa_R = self._get_kappa_R(polyno, wndwlen)

        fkappa = interp1d(self.Rbin_centers, kappa_R, bounds_error=False, fill_value=(np.nan, np.nan))
        R_grid = np.sqrt(self.xbin_centers_2d**2 + self.ybin_centers_2d**2)
        return fkappa(R_grid)

    ###----------- HR-only reliable features, LR cubes -----------###

    def get_SFR_surfdens_xy(self, PartType: int=0) -> np.array:
        '''Get the 2D surface density of the star formation rate in cgs'''
        SFR_surfdens, _, _, _ = binned_statistic_2d(
            self.data[PartType]["x_coords"], self.data[PartType]["y_coords"], self.data[PartType]["SFRs"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return SFR_surfdens / (self.xybin_width * self.xybin_width)

    def get_SFR_voldens_xyz(self, PartType: int=0) -> np.array:
        '''Gas star formation rate volume density in cgs units'''
        SFR_densities = np.zeros((self.xybinno, self.xybinno, self.zbinno_ptl)) * np.nan
        for zbmin, zbmax, k in zip(self.zbin_edges_ptl[:-1], self.zbin_edges_ptl[1:], range(self.zbinno_ptl)):
            cnd = (self.data[PartType]["z_coords"] > zbmin) & (self.data[PartType]["z_coords"] < zbmax)
            if len(self.data[PartType]["x_coords"][cnd]) == 0:
                SFR_densities[:,:,k] = np.zeros((self.xybinno, self.xybinno)) * np.nan
                continue
            dens, _, _, _ = binned_statistic_2d(
                self.data[PartType]["x_coords"][cnd], self.data[PartType]["y_coords"][cnd], self.data[PartType]["SFRs"][cnd],
                bins=(self.xbin_edges, self.ybin_edges),
                statistic='sum'
            )
            SFR_densities[:,:,k] = dens / (self.xybin_width * self.xybin_width * self.zbin_width_ptl)
        return SFR_densities
    
    def get_H2_mass_frac_xy(self, PartType: int=0) -> np.array:
        '''Get the H2 mass fraction per 2D column, dimensionless'''
        H2_frac, _, _, _ = binned_statistic_2d(
            self.data[PartType]["x_coords"], self.data[PartType]["y_coords"], self.data[PartType]["masses"]*self.data[PartType]["xH2"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        mass, _, _, _ = binned_statistic_2d(
            self.data[PartType]["x_coords"], self.data[PartType]["y_coords"], self.data[PartType]["masses"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return H2_frac / mass

    def get_H2_mass_frac_xyz(self, PartType: int=0) -> np.array:
        '''Get the H2 mass fraction, dimensionless'''
        H2_frac = np.zeros((self.xybinno, self.xybinno, self.zbinno)) * np.nan
        for zbmin, zbmax, k in zip(self.zbin_edges[:-1], self.zbin_edges[1:], range(self.zbinno)):
            cnd = (self.data[PartType]["z_coords"] > zbmin) & (self.data[PartType]["z_coords"] < zbmax)
            if len(self.data[PartType]["x_coords"][cnd]) == 0:
                H2_frac[:,:,k] = np.zeros((self.xybinno, self.xybinno)) * np.nan
                continue
            H2mass, _, _, _ = binned_statistic_2d(
                self.data[PartType]["x_coords"][cnd], self.data[PartType]["y_coords"][cnd], self.data[PartType]["masses"][cnd]*self.data[PartType]["xH2"],
                bins=(self.xbin_edges, self.ybin_edges),
                statistic='sum'
            )
            mass, _, _, _ = binned_statistic_2d(
                self.data[PartType]["x_coords"][cnd], self.data[PartType]["y_coords"][cnd], self.data[PartType]["masses"][cnd]*self.data[PartType]["xH2"],
                bins=(self.xbin_edges, self.ybin_edges),
                statistic='sum'
            )
            H2_frac[:,:,k] = H2mass / mass
        return H2_frac
    
    def get_HI_mass_frac_xy(self, PartType: int=0) -> np.array:
        '''Get the HI mass fraction per 2D column, dimensionless'''
        HI_frac, _, _, _ = binned_statistic_2d(
            self.data[PartType]["x_coords"], self.data[PartType]["y_coords"], self.data[PartType]["masses"]*self.data[PartType]["xHI"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        mass, _, _, _ = binned_statistic_2d(
            self.data[PartType]["x_coords"], self.data[PartType]["y_coords"], self.data[PartType]["masses"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return HI_frac / mass

    def get_HI_mass_frac_xyz(self, PartType: int=0) -> np.array:
        '''Get the HI mass fraction, dimensionless'''
        HI_frac = np.zeros((self.xybinno, self.xybinno, self.zbinno)) * np.nan
        for zbmin, zbmax, k in zip(self.zbin_edges[:-1], self.zbin_edges[1:], range(self.zbinno)):
            cnd = (self.data[PartType]["z_coords"] > zbmin) & (self.data[PartType]["z_coords"] < zbmax)
            if len(self.data[PartType]["R_coords"][cnd]) == 0:
                HI_frac[:,:,k] = np.zeros((self.xybinno, self.xybinno)) * np.nan
                continue
            HImass, _, _, _ = binned_statistic_2d(
                self.data[PartType]["x_coords"][cnd], self.data[PartType]["y_coords"][cnd], self.data[PartType]["masses"][cnd]*self.data[PartType]["xHI"],
                bins=(self.xbin_edges, self.ybin_edges),
                statistic='sum'
            )
            mass, _, _, _ = binned_statistic_2d(
                self.data[PartType]["x_coords"][cnd], self.data[PartType]["y_coords"][cnd], self.data[PartType]["masses"][cnd],
                bins=(self.xbin_edges, self.ybin_edges),
                statistic='sum'
            )
            HI_frac[:,:,k] = HImass / mass
        return HI_frac

    def get_density_xyz(self, zbinwidth: float=None, zbinedges: np.array=None, zbinno: int=None, PartType: int=None, SFR_cnd: bool=False) -> np.array:
        '''Get the 3D volume density in cgs. If SFR_cnd==True, only uses star-forming gas.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_density_xyz.")
        if zbinwidth==None:
            zbinwidth = self.zbin_width_ptl
            zbinedges = self.zbin_edges_ptl
            zbinno = self.zbinno_ptl

        densities = np.zeros((self.xybinno, self.xybinno, zbinno)) * np.nan
        for zbmin, zbmax, k in zip(zbinedges[:-1], zbinedges[1:], range(zbinno)):
            cnd = (self.data[PartType]["z_coords"] > zbmin) & (self.data[PartType]["z_coords"] < zbmax)
            if SFR_cnd:
                cnd = cnd & (self.data[PartType]["SFRs"] > 0)
            if len(self.data[PartType]["R_coords"][cnd]) == 0:
                densities[:,:,k] = np.zeros((self.xybinno, self.xybinno)) * np.nan
                continue
            dens, _, _, _ = binned_statistic_2d(
                self.data[PartType]["x_coords"][cnd], self.data[PartType]["y_coords"][cnd], self.data[PartType]["masses"][cnd],
                bins=(self.xbin_edges, self.ybin_edges),
                statistic='sum'
            )
            densities[:,:,k] = dens / (self.xybin_width * self.xybin_width * zbinwidth)
        return densities
    
    def get_potential_xyz(
        self,
        PartTypes: List[int] = [0,1,2,3,4],
        eps: float=1., # shape parameter, defaults to 1.
        kernel: str="cubic", # 3rd-order polyharmonic spline
        neighbors: int=64 # number of neighbors to use for the interpolation
    ):
        '''Get the potential array from the gas cells, in cgs units. This uses the RBF
        interpolation class from scipy.'''

        try:
            x_all = np.concatenate([self.data[i]["x_coords"] for i in PartTypes if self.data[i] is not None])
            y_all = np.concatenate([self.data[i]["y_coords"] for i in PartTypes if self.data[i] is not None])
            z_all = np.concatenate([self.data[i]["z_coords"] for i in PartTypes if self.data[i] is not None])
            ptl_all = np.concatenate([self.data[i]["Potential"] for i in PartTypes if self.data[i] is not None])
            coords_all = np.array([x_all, y_all, z_all]).T
        except KeyError:
            logger.critical("Requested particle types not loaded: {:s}".format(str([i for i in PartTypes if i not in self.data])))
            sys.exit(1)

        interp = RBFInterpolator(
            coords_all,
            ptl_all,
            kernel=kernel,
            epsilon=eps,
            neighbors=neighbors
        )

        coords_interp = np.array([self.xbin_centers_3d_ptl.flatten(), self.ybin_centers_3d_ptl.flatten(), self.zbin_centers_3d_ptl.flatten()]).T
        return interp(coords_interp).reshape(self.xybinno, self.xybinno, self.zbinno_ptl)

    def _set_weight_integrand_xyz(self, PartTypes: List[int] = [0,1,2,3,4], PartType: int=None) -> np.array:
        '''Get the integrand for the weight function, in cgs units.'''

        rho_grid = self.get_density_xyz(PartType=PartType)
        ptl_grid = self.get_potential_xyz(PartTypes=PartTypes)

        dz = np.gradient(self.zbin_centers_3d_ptl, axis=2)
        dPhi = np.gradient(ptl_grid, axis=2)
        dPhidz = dPhi/dz

        self._weight_integrand = rho_grid * dPhidz * self.zbin_width_ptl

    def get_weight_xy(self, PartTypes: List[int] = [0,1,2,3,4], PartType: int=None) -> np.array:
        '''Get the weights for the interstellar medium, based on the density and potential
        grids, assuming the potential is symmetrical about the mid-plane of the disk.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_weight_xy (this is used for computing\
                            the density, all gas is used for the potential).")

        if self._weight_integrand is None:
            self._set_weight_integrand_xyz(PartTypes=PartTypes, PartType=PartType)
        return np.nansum(np.fabs(self._weight_integrand)/2., axis=2)

    def get_int_force_left_right_xy(self, PartTypes: List[int] = [0,1,2,3,4], PartType: int=None) -> Tuple[np.array, np.array, np.array]:
        '''Get integrated force per unit area, separated into its components above and below
        the mid-plane of the disk.'''

        if self._weight_integrand is None:
            self._set_weight_integrand_xyz(PartTypes=PartTypes, PartType=PartType)
        z_mp_idcs = np.nanargmin(np.nancumsum(self._weight_integrand, axis=2), axis=2)

        integrand_left = np.zeros_like(self._weight_integrand)
        integrand_right = np.zeros_like(self._weight_integrand)
        for i in range(self.xybinno):
            for j in range(self.xybinno):
                integrand_left[i,j,:z_mp_idcs[i,j]] = self._weight_integrand[i,j,:z_mp_idcs[i,j]]
                integrand_right[i,j,z_mp_idcs[i,j]:] = self._weight_integrand[i,j,z_mp_idcs[i,j]:]
                
        return np.nansum(integrand_left, axis=2), np.nansum(integrand_right, axis=2), z_mp_idcs

    def get_force_xy(self, PartTypes: List[int] = [0,1,2,3,4], PartType: int=6) -> np.array:
        '''Get the net force resulting from an asymmetric potential, in cgs units.'''

        if self._weight_integrand is None:
            self._set_weight_integrand_xyz(PartTypes=PartTypes, PartType=PartType)
        return np.nansum(self._weight_integrand, axis=2)

    ###----------- HR-only reliable features, mid-plane -----------###

    def get_midplane_density_xy(self, PartType: int=None) -> np.array:
        '''Get the 3D mid-plane density in cgs.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_midplane_density_xy.")

        dens_3D = self.get_density_xyz(PartType=PartType)
        if self.midplane_idcs is not None:
            return self._select_predef_midplane(dens_3D)
        else: # estimate the mid-plane by returning the maximum value along the z-axis
            return np.nanmax(dens_3D, axis=2)

    def _get_gas_av_vel_xyz_xy(
        self,
        z_min: float=None,
        z_max: float=None,
        PartType: int=None
    ) -> Tuple[np.array, np.array, np.array]:
        '''Get the average 2D gas velocity components in the x, y, and z directions, in cm/s.'''

        if PartType==None:
            logger.critical("Please specify a particle type for _get_gas_av_vel_xyz_xy.")
        if PartType!=0 and PartType!=6:
            logger.critical("Please set PartType0 or PartType 6 in _get_gas_av_vel_xyz_xy (gas particles only).")

        if z_min==None:
            z_min = -self.total_height
        if z_max==None:
            z_max = self.total_height

        cnd = (self.data[PartType]["z_coords"] > z_min) & (self.data[PartType]["z_coords"] < z_max)
        if len(self.data[PartType]["x_coords"][cnd]) == 0:
            logger.critical("No gas cells found in the given z-range.")

        summass, _, _, _ = binned_statistic_2d(
            self.data[PartType]["x_coords"][cnd], self.data[PartType]["y_coords"][cnd], self.data[PartType]["masses"][cnd],
            bins=(self.xbin_edges, self.ybin_edges), expand_binnumbers=True,
            statistic='sum'
        )

        meanvels = []
        for velstring in ["velxs", "velys", "velzs"]:
            meanvel, _, _, _ = binned_statistic_2d(
                self.data[PartType]["x_coords"][cnd], self.data[PartType]["y_coords"][cnd], self.data[PartType]["masses"][cnd]*self.data[PartType][velstring][cnd],
                bins=(self.xbin_edges, self.ybin_edges), expand_binnumbers=True,
                statistic='sum'
            )
            meanvel /= summass
            meanvels.append(meanvel)

        return tuple(meanvels)

    def _get_gas_veldisps_xyz_xy(
        self,
        meanvels_xyz: Tuple=None,
        z_min: float=None,
        z_max: float=None,
        PartType: int=None,
    ) -> Tuple[np.array, np.array, np.array]:
        '''2D Gas velocity dispersion components in cgs. Distinct from the mid-plane turbulent
        velocity dispersion, this is the velocity dispersion along columns in z.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_gas_veldisps_xyz_xy.")
        if PartType!=0 and PartType!=6:
            logger.critical("Please set PartType0 or PartType 6 in get_gas_veldisps_xyz_xy (gas particles only).")

        if z_min==None:
            z_min = -self.total_height
        if z_max==None:
            z_max = self.total_height

        cnd = (self.data[PartType]["z_coords"] > z_min) & (self.data[PartType]["z_coords"] < z_max)
        if len(self.data[PartType]["x_coords"][cnd]) == 0:
            logger.critical("No gas cells found in the given z-range.")

        summass, x_edge, y_edge, binnumbers = binned_statistic_2d(
            self.data[PartType]["x_coords"][cnd], self.data[PartType]["y_coords"][cnd], self.data[PartType]["masses"][cnd],
            bins=(self.xbin_edges, self.ybin_edges), expand_binnumbers=True,
            statistic='sum'
        )

        '''Subtract the mean velocity in each bin from the velocity components, then take the
        root-mean-square of the differences to get the velocity dispersions.'''
        if meanvels_xyz==None:
            meanvels = self._get_gas_av_vel_xyz_xy(z_min=z_min, z_max=z_max, PartType=PartType)
        else:
            meanvels = meanvels_xyz

        maxbinnumber_x = len(x_edge)-1
        maxbinnumber_y = len(y_edge)-1
        veldisps_xyz = []
        for velstring, meanvel in zip(["velxs", "velys", "velzs"], meanvels):
            bn_x, bn_y = binnumbers.copy()
            bn_x[bn_x>maxbinnumber_x] = maxbinnumber_x
            bn_y[bn_y>maxbinnumber_y] = maxbinnumber_y
            bn_x[bn_x<1] = 1
            bn_y[bn_y<1] = 1
            bn_x -= 1
            bn_y -= 1
            vel_minus_mean = self.data[PartType][velstring][cnd]-meanvel[bn_x, bn_y]

            sumveldisp, _, _, _ = binned_statistic_2d(
                self.data[PartType]["x_coords"][cnd], self.data[PartType]["y_coords"][cnd], self.data[PartType]["masses"][cnd]*vel_minus_mean**2,
                bins=(self.xbin_edges, self.ybin_edges),
                statistic='sum'
            )
            veldisps_xyz.append(np.sqrt(sumveldisp/summass))

        return tuple(veldisps_xyz)

    def get_gas_midplane_veldisps_xyz_xy(
        self,
        PartType: int=None,
    ) -> Tuple[np.array, np.array, np.array]:
        meanvels = self._get_gas_av_vel_xyz_xy(z_min=-self.total_height, z_max=self.total_height, PartType=PartType)
        veldisps_x = np.zeros((self.xybinno, self.xybinno, self.zbinno_ptl)) * np.nan
        veldisps_y = np.zeros((self.xybinno, self.xybinno, self.zbinno_ptl)) * np.nan
        veldisps_z = np.zeros((self.xybinno, self.xybinno, self.zbinno_ptl)) * np.nan
        meanvels = self._get_gas_av_vel_xyz_xy(z_min=-self.total_height, z_max=self.total_height, PartType=PartType)
        for zbmin, zbmax, k in zip(self.zbin_edges_ptl[:-1], self.zbin_edges_ptl[1:], range(self.zbinno_ptl)):
            cnd = (self.data[PartType]["z_coords"] > zbmin) & (self.data[PartType]["z_coords"] < zbmax)
            if len(self.data[PartType]["x_coords"][cnd]) == 0:
                veldisps_z[:,:,k] = np.ones((self.xybinno, self.xybinno)) * np.nan
                continue
            veldisp_x, veldisp_y, veldisp_z = self._get_gas_veldisps_xyz_xy(
                meanvels_xyz=meanvels, z_min=zbmin, z_max=zbmax, PartType=PartType)
            veldisps_x[:,:,k] = veldisp_x
            veldisps_y[:,:,k] = veldisp_y
            veldisps_z[:,:,k] = veldisp_z

        if self.midplane_idcs is not None:
            return self._select_predef_midplane(veldisps_x), self._select_predef_midplane(veldisps_y), self._select_predef_midplane(veldisps_z)
        else: # estimate the mid-plane by returning the maximum value along the z-axis
            return np.nanmax(veldisps_x, axis=2), np.nanmax(veldisps_y, axis=2), np.nanmax(veldisps_z, axis=2)

    def _get_gas_turbpress_xyz(self, PartType: int=None) -> np.array:
        '''3D Gas turbulent pressure in cgs units, divided by the Boltzmann constant.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_gas_turbpress_xyz.")
        if PartType!=0 and PartType!=6:
            logger.critical("Please set PartType0 or PartType 6 in get_gas_turbpress_xyz (gas particles only).")

        density = self.get_density_xyz(PartType=PartType)

        veldisps_z = np.zeros((self.xybinno, self.xybinno, self.zbinno_ptl)) * np.nan
        meanvels = self._get_gas_av_vel_xyz_xy(z_min=-self.total_height, z_max=self.total_height, PartType=PartType)
        for zbmin, zbmax, k in zip(self.zbin_edges_ptl[:-1], self.zbin_edges_ptl[1:], range(self.zbinno_ptl)):
            cnd = (self.data[PartType]["z_coords"] > zbmin) & (self.data[PartType]["z_coords"] < zbmax)
            if len(self.data[PartType]["x_coords"][cnd]) == 0:
                veldisps_z[:,:,k] = np.ones((self.xybinno, self.xybinno)) * np.nan
                continue
            _, _, veldisp_z = self._get_gas_veldisps_xyz_xy(meanvels_xyz=meanvels, z_min=zbmin, z_max=zbmax, PartType=PartType)
            veldisps_z[:,:,k] = veldisp_z
        
        turbpress_3D = veldisps_z**2 * density / ah.kB_cgs

        return turbpress_3D

    def get_gas_midplane_turbpress_xy(self, PartType: int=None) -> np.array:
        '''Mid-plane gas turbulent pressure (2D) in the vertical/plane-perpendicular direction, in cgs units,
        divided by the Boltzmann constant.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_gas_midplane_turbpress_xy.")
        if PartType!=0 and PartType!=6:
            logger.critical("Please set PartType0 or PartType 6 in get_gas_midplane_turbpress_xy (gas particles only).")

        turbpress_3D = self._get_gas_turbpress_xyz(PartType=PartType)
        if self.midplane_idcs is not None:
            return self._select_predef_midplane(turbpress_3D)
        else: # estimate the mid-plane by returning the maximum value along the z-axis
            return np.nanmax(turbpress_3D, axis=2)

    def get_gas_midplane_thermpress_xy(self, PartType: int=None) -> np.array:
        '''Gas midplane thermal pressure (2D) in cgs units.'''

        if PartType==None:
            logger.critical("Please specify a particle type for get_gas_midplane_thermpress_xy.")
        if PartType!=0 and PartType!=6:
            logger.critical("Please set PartType0 or PartType 6 in get_gas_midplane_thermpress_xy (gas particles only).")

        Pth = np.zeros((self.xybinno, self.xybinno, self.zbinno_ptl)) * np.nan
        for zbmin, zbmax, k in zip(self.zbin_edges_ptl[:-1], self.zbin_edges_ptl[1:], range(self.zbinno_ptl)):
            cnd = (self.data[PartType]["z_coords"] > zbmin) & (self.data[PartType]["z_coords"] < zbmax)
            if len(self.data[PartType]["x_coords"][cnd]) == 0:
                Pth[:,:,k] = np.ones((self.xybinno, self.xybinno)) * np.nan
                continue
            Pth_bin, _, _, _ = binned_statistic_2d(
                self.data[PartType]["x_coords"][cnd], self.data[PartType]["y_coords"][cnd], self.data[PartType]["masses"][cnd]*self.data[PartType]["U"][cnd],
                bins=(self.xbin_edges, self.ybin_edges),
                statistic='sum'
            )
            Pth[:,:,k] = Pth_bin / (self.xybin_width * self.xybin_width * self.zbin_width_ptl) # mass * U --> dens * U

        if self.midplane_idcs is not None:
            return self._select_predef_midplane(Pth * (ah.gamma-1.) / ah.kB_cgs)
        else:
            return np.nanmax(Pth * (ah.gamma-1.) / ah.kB_cgs, axis=2) # dens * U --> Ptherm