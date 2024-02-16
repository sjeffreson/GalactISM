from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np

from scipy.signal import savgol_filter as sg
from scipy.stats import binned_statistic_2d
from scipy.interpolate import interp1d

import h5py
import astro_helper as ah
from invdisttree import Invdisttree

DEFAULT_ROOT_DIR = Path("/n/holystore01/LABS/itc_lab/Users/sjeffreson/")
class GriddedDataset:
    '''Grid Voronoi cells into a 3D array of desired resolution and calculate
    relevant training data for the normalizing flows, at this resolution.'''

    def __init__(
        self,
        galaxy_type: str,
        max_grid_width: float = 30., # kpc, generally set to galaxy diameter
        max_grid_height: float = 1.5, # kpc
        resolution: float = 80., # pc
        ptl_resolution: float = None, # pc, for computation of the potential
        root_dir: Path = DEFAULT_ROOT_DIR,
        snapname: str = "snap-DESPOTIC_300.hdf5",
        realign_galaxy: bool=True, # according to angular momentum vector of gas
    ):
        
        self.galaxy_type = galaxy_type
        self.max_grid_width = max_grid_width * ah.kpc_to_cm
        self.max_grid_height = max_grid_height * ah.kpc_to_cm
        self.resolution = resolution * ah.pc_to_cm
        self.ptl_resolution = ptl_resolution
        if self.ptl_resolution != None:
            self.ptl_resolution = self.ptl_resolution * ah.pc_to_cm
        self.root_dir = root_dir
        self.snapname = snapname
        self.realign_galaxy = realign_galaxy

        self.gas_data = self.read_snap_data(0)
        if realign_galaxy:
            self.x_CM, self.y_CM, self.z_CM, self.vx_CM, self.vy_CM, self.vz_CM = self.get_gas_disk_COM()
            self.Lx, self.Ly, self.Lz = self.get_gas_disk_angmom()
            self.gas_data = self.set_realign_galaxy(self.gas_data)

        # (degraded) grid on which to compute arrays
        # TODO: Replace inverse distance interpolation with RBF interpolation package
        # TODO: Replace gridding in general with RBF interpolation package, right now it's just nearest neighbor
        self.xy_binnum = int(np.ceil(self.max_grid_width/self.resolution))
        self.xbin_edges = np.linspace(-self.max_grid_width, self.max_grid_width, self.xy_binnum+1)
        self.ybin_edges = np.linspace(-self.max_grid_width, self.max_grid_width, self.xy_binnum+1)
        self.xbin_centers = (self.xbin_edges[1:]+self.xbin_edges[:-1])*0.5
        self.ybin_centers = (self.ybin_edges[1:]+self.ybin_edges[:-1])*0.5

        self.x_grid, self.y_grid = np.meshgrid(self.xbin_centers, self.ybin_centers)
    
    def get_grid(self) -> Tuple[np.array, np.array, np.array]:
        return self.x_grid, self.y_grid

    def read_snap_data(
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
        snapshot = h5py.File(DEFAULT_ROOT_DIR / self.galaxy_type / self.snapname, "r")
        header = snapshot["Header"]
        if "PartType"+str(PartType) not in snapshot:
            return None
        else:
            PartType_data = snapshot["PartType"+str(PartType)]

        snap_data = {}
        snap_data["x_coords"] = (PartType_data['Coordinates'][:,0] - 0.5 * header.attrs['BoxSize']) * PartType_data['Coordinates'].attrs['to_cgs']
        snap_data["y_coords"] = (PartType_data['Coordinates'][:,1] - 0.5 * header.attrs['BoxSize']) * PartType_data['Coordinates'].attrs['to_cgs']
        snap_data["R_coords"] = np.sqrt(snap_data["x_coords"]**2 + snap_data["y_coords"]**2)
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
            return snap_data
        else:
            snap_data["Density"] = PartType_data['Density'][:] * PartType_data['Density'].attrs['to_cgs']
            snap_data["U"] = PartType_data['InternalEnergy'][:] * PartType_data['InternalEnergy'].attrs['to_cgs']
            snap_data["voldenses"] = PartType_data['Density'][:] * PartType_data['Density'].attrs['to_cgs']
            snap_data["temps"] = (ah.gamma - 1.) * snap_data["U"] / ah.kB_cgs * ah.mu * ah.mp_cgs
            snap_data["SFRs"] = PartType_data['StarFormationRate'][:] * PartType_data['StarFormationRate'].attrs['to_cgs']
            snap_data["AlphaVir"] = PartType_data['AlphaVir'][:]
            return snap_data
    
    def cut_out_gas_disk(self) -> Dict[str, np.array]:
        '''cut out most of the gas cells that are in the background grid, not the disk'''
        cnd = (self.gas_data["R_coords"] < self.max_grid_width/2.) & (np.fabs(self.gas_data["z_coords"]) < self.max_grid_width/2.)
        return {key: value[cnd] for key, value in self.gas_data.items()}

    def get_gas_disk_COM(self) -> Tuple[float, float, float, float, float, float]:
        '''Get the center of mass (positional and velocity) of the gas disk'''
        gas_data_cut = self.cut_out_gas_disk()

        x_CM = np.average(gas_data_cut["x_coords"], weights=gas_data_cut["masses"])
        y_CM = np.average(gas_data_cut["y_coords"], weights=gas_data_cut["masses"])
        z_CM = np.average(gas_data_cut["z_coords"], weights=gas_data_cut["masses"])
        vx_CM = np.average(gas_data_cut["velxs"], weights=gas_data_cut["masses"])
        vy_CM = np.average(gas_data_cut["velys"], weights=gas_data_cut["masses"])
        vz_CM = np.average(gas_data_cut["velzs"], weights=gas_data_cut["masses"])

        return x_CM, y_CM, z_CM, vx_CM, vy_CM, vz_CM

    def get_gas_disk_angmom(self) -> Tuple[float, float, float]:
        '''Get the angular momentum vector of the gas disk'''
        x_CM, y_CM, z_CM, vx_CM, vy_CM, vz_CM = self.get_gas_disk_COM()
        gas_data_cut = self.cut_out_gas_disk()

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

    def set_realign_galaxy(self, snap_data: Dict[str, np.array]) -> Dict[str, np.array]:
        '''Realign the galaxy according to the center of mass and the angular momentum
        vector of the gas disk'''

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

        return snap_data

    def get_rotation_curve(self, deltaR: float=10.) -> Tuple[np.array, np.array]:
        '''Get the rotation curve of the galaxy, in cm/s. DeltaR is the size of the radial bins
        (pc) used to compute the rotation curve at all azimuthal angles, which is then interpolated
        to obtain the grid.'''
        Rbin_edges = np.linspace(0., self.max_grid_width, int(np.rint(self.max_grid_width/(deltaR*ah.pc_to_cm))))
        Rbin_centres = (Rbin_edges[1:]+Rbin_edges[:-1])/2.
        vcs = []

        for Rbmin, Rbmax in zip(Rbin_edges[:-1], Rbin_edges[1:]):
            cnd = (self.gas_data['R_coords'] > Rbmin) & (self.gas_data['R_coords'] < Rbmax)
            if(len(self.gas_data['R_coords'][cnd])>0):
                vcs.append(np.average(
                    -self.gas_data['y_coords'][cnd]/self.gas_data['R_coords'][cnd] * self.gas_data['velxs'][cnd] +
                    self.gas_data['x_coords'][cnd]/self.gas_data['R_coords'][cnd] * self.gas_data['velys'][cnd],
                    weights=self.gas_data['masses'][cnd]))
            else:
                vcs.append(np.nan)

        return Rbin_centres, vcs

    def get_Omegaz_array(self, deltaR: float=10.) -> np.array:
        '''Get the galactic angular velocity in the z-direction, in /s.'''
        Rbin_centres, vcs = self.get_rotation_curve(deltaR=deltaR)
        Omegazs = vcs / Rbin_centres

        fOmegaz = interp1d(Rbin_centres, vcs, bounds_error=False, fill_value=(np.nan, np.nan))
        R_grid = np.sqrt(self.x_grid**2 + self.y_grid**2)
        return fOmegaz(R_grid)
    
    def get_kappa_array(self, deltaR: float=10., polyno: int=2, wndwlen: int=9) -> np.array:
        '''Get the epicyclic frequency in the z-direction, in /s.'''
        Rbin_centres, vcs = self.get_rotation_curve(deltaR=deltaR)
        Omegazs = vcs / Rbin_centres

        dR = sg(Rbin_centres, wndwlen, polyno, deriv=1)
        dvc = sg(vcs, wndwlen, polyno, deriv=1)
        betas = dvc/dR * Rbin_centres/vcs
        kappas = Omegazs * np.sqrt(2.*(1.+betas))

        fkappa = interp1d(Rbin_centres, kappas, bounds_error=False, fill_value=(np.nan, np.nan))
        R_grid = np.sqrt(self.x_grid**2 + self.y_grid**2)
        return fkappa(R_grid)

    def get_potential_z_grid(self) -> Tuple[int, np.array]:
        ptl_z_binnum = int(np.ceil(2.*self.max_grid_height/self.ptl_resolution))
        ptl_z_bin_edges = np.linspace(-self.max_grid_height, self.max_grid_height, ptl_z_binnum+1)
        ptl_z_bin_centers = (ptl_z_bin_edges[1:]+ptl_z_bin_edges[:-1])*0.5
        return ptl_z_binnum, ptl_z_bin_edges, ptl_z_bin_centers

    def get_potential_array(self, leafsize: int=10, eps: float=6., p: int=1) -> np.array:
        '''Get the potential array from the gas cells, in cgs units. Leafsize, eps and p
        are performance parameters for the inverse distance interpolation. See description
        in invdisttree.py'''

        if self.ptl_resolution == None:
            raise ValueError("The vertical resolution for computation of the potential must be set to use this function.")

        '''the z-grid for calculation of the potential needs to be sufficiently-fine to capture
        density fluctations near the simulation resolution'''
        ptl_z_binnum, ptl_z_bin_edges, ptl_z_bin_centers = self.get_potential_z_grid()

        snapdata_all = [self.read_snap_data(i) for i in range(5) if self.read_snap_data(i) != None]
        snapdata_all = [self.set_realign_galaxy(snapdata) for snapdata in snapdata_all]
        x_all = np.concatenate([snapdata["x_coords"] for snapdata in snapdata_all])
        y_all = np.concatenate([snapdata["y_coords"] for snapdata in snapdata_all])
        z_all = np.concatenate([snapdata["z_coords"] for snapdata in snapdata_all])
        ptl_all = np.concatenate([snapdata["Potential"] for snapdata in snapdata_all])

        x_grid, y_grid, z_grid = np.meshgrid(self.xbin_centers, self.ybin_centers, ptl_z_bin_centers)

        invdisttree = Invdisttree(np.array([x_all, y_all, z_all]).T, ptl_all, leafsize=leafsize, stat=1)
        ptl_grid = invdisttree(np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()]).T, nnear=8, eps=eps, p=p)
        ptl_grid = np.reshape(ptl_grid, np.shape(x_grid))

        return ptl_grid

    def get_rho_potential_array(self) -> np.array:
        '''Get the density array corresponding to the potential grid, for computation of
        interstellar medium weights, in cgs units.'''

        if self.ptl_resolution == None:
            raise ValueError("The vertical resolution for computation of the potential must be set to use this function.")

        ptl_z_binnum, ptl_z_bin_edges, ptl_z_bin_centers = self.get_potential_z_grid()

        cnd = (self.gas_data["AlphaVir"] > 2.) | (self.gas_data["AlphaVir"] <= 0.) # cut out gas that's in gravitationally-bound clouds
        x = self.gas_data["x_coords"][cnd]
        y = self.gas_data["y_coords"][cnd]
        z = self.gas_data["z_coords"][cnd]
        mass = self.gas_data["masses"][cnd]

        rho3Ds = np.zeros((self.xy_binnum, self.xy_binnum, ptl_z_binnum))
        for zbmin, zbmax, k in zip(ptl_z_bin_edges[:-1], ptl_z_bin_edges[1:], range(ptl_z_binnum)):
            cnd = (z > zbmin) & (z < zbmax)
            mass_bin = mass[cnd]
            x_bin = x[cnd]
            y_bin = y[cnd]

            masssum, xedges, yedges, binnumber = binned_statistic_2d(
                x_bin, y_bin,
                mass_bin,
                bins=[self.xbin_edges, self.ybin_edges],
                statistic='sum')
            rho3Ds[:,:,k] = masssum/self.resolution/self.resolution/self.ptl_resolution
        
        return rho3Ds

    def get_weight_array(self, polyno: int=2, wndwlen: int=9) -> np.array:
        '''Get the weights for the interstellar medium, based on the density and potential
        grids. Polyno and wndlen are the parameters for the Savitzky-Golay filter, which
        is used to take the z-derivative of the potential grid. Output in cgs units.'''

        if self.ptl_resolution == None:
            raise ValueError("The vertical resolution for computation of the potential must be set to use this function.")

        ptl_z_binnum, ptl_z_bin_edges, ptl_z_bin_centers = self.get_potential_z_grid()
        deltaz = ptl_z_bin_centers[1]-ptl_z_bin_centers[0]
        x_grid, y_grid, z_grid = np.meshgrid(self.xbin_centers, self.ybin_centers, ptl_z_bin_centers)

        rho_grid = self.get_rho_potential_array()
        ptl_grid = self.get_potential_array()
        
        dz = sg(z_grid, wndwlen, polyno, deriv=1, axis=2)
        dPhi = sg(ptl_grid, wndwlen, polyno, deriv=1, axis=2)
        dPhidz = dPhi/dz

        return np.sum(np.fabs(rho_grid*dPhidz*deltaz)/2., axis=2)

    def get_gas_surfdens_array(self) -> np.array:
        '''Gas surface density in solar masses per pc^2'''
        surfdens, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["masses"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return surfdens/ah.Msol_to_g/(self.resolution/ah.pc_to_cm)**2

    def get_stellar_surfdens_array(self) -> np.array:
        '''Stellar surface density in solar masses per pc^2'''
        stardata_all = [self.read_snap_data(i) for i in [2,3,4] if self.read_snap_data(i) != None]
        stardata_all = [self.set_realign_galaxy(stardata) for stardata in stardata_all]
        x_stars = np.concatenate([stardata["x_coords"] for stardata in stardata_all])
        y_stars = np.concatenate([stardata["y_coords"] for stardata in stardata_all])
        z_stars = np.concatenate([stardata["z_coords"] for stardata in stardata_all])
        mass_stars = np.concatenate([stardata["masses"] for stardata in stardata_all])

        surfdens, x_edge, y_edge, binnumber = binned_statistic_2d(
            x_stars, y_stars, mass_stars,
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return surfdens/ah.Msol_to_g/(self.resolution/ah.pc_to_cm)**2

    def get_SFR_surfdens_array(self) -> np.array:
        '''Gas star formation rate surface density in solar masses per kpc^2 per yr'''
        SFR, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["SFRs"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return SFR/ah.Msol_to_g/(self.resolution/ah.kpc_to_cm)**2*ah.yr_to_s

    def get_midplane_gas_dens_array(self) -> np.array:
        '''Gas volume density in solar masses per pc^3'''
        dens, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["masses"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return dens/ah.Msol_to_g/(self.resolution/ah.pc_to_cm)**3

    def get_midplane_stellar_dens_array(self) -> np.array:
        '''Stellar volume density in solar masses per pc^3'''
        stardata_all = [self.read_snap_data(i) for i in [2,3,4] if self.read_snap_data(i) != None]
        stardata_all = [self.set_realign_galaxy(stardata) for stardata in stardata_all]
        x_stars = np.concatenate([stardata["x_coords"] for stardata in stardata_all])
        y_stars = np.concatenate([stardata["y_coords"] for stardata in stardata_all])
        z_stars = np.concatenate([stardata["z_coords"] for stardata in stardata_all])
        mass_stars = np.concatenate([stardata["masses"] for stardata in stardata_all])

        dens, x_edge, y_edge, binnumber = binned_statistic_2d(
            x_stars, y_stars, mass_stars,
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return dens/ah.Msol_to_g/(self.resolution/ah.pc_to_cm)**3

    def get_midplane_SFR_dens_array(self) -> np.array:
        '''Gas star formation rate volume density in solar masses per kpc^3 per yr'''
        Sfr, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["SFRs"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return Sfr/ah.Msol_to_g/(self.resolution/ah.kpc_to_cm)**3*ah.yr_to_s

    def get_gas_veldisps_xyz_array(self) -> Tuple[np.array, np.array, np.array]:
        '''Gas velocity dispersion components in km/s'''
        sumrhos, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["masses"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )

        veldisps_xyz = []
        maxbinnumber = int(np.rint(self.max_grid_width/self.resolution))
        for velstring in ["velxs", "velys", "velzs"]:
            summeanvel, x_edge, y_edge, binnumber = binned_statistic_2d(
                self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["masses"]*self.gas_data[velstring],
                bins=(self.xbin_edges, self.ybin_edges), expand_binnumbers=True,
                statistic='sum'
            )
            meanvel = summeanvel/sumrhos

            bn_x, bn_y = binnumber
            bn_x[bn_x>maxbinnumber] = maxbinnumber
            bn_y[bn_y>maxbinnumber] = maxbinnumber
            bn_x[bn_x<1] = 1
            bn_y[bn_y<1] = 1
            bn_x -= 1
            bn_y -= 1
            vel_minus_mean = self.gas_data[velstring]-meanvel[bn_x, bn_y]

            sumveldisp, x_edge, y_edge, binnumber = binned_statistic_2d(
                self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["masses"]*vel_minus_mean**2,
                bins=(self.xbin_edges, self.ybin_edges),
                statistic='sum'
            )
            veldisps_xyz.append(np.sqrt(sumveldisp/sumrhos)/ah.kms_to_cms)

        return tuple(veldisps_xyz)

    def get_gas_turbpress_array(self) -> np.array:
        '''Gas turbulent pressure in cgs units'''
        veldisps_xyz = self.get_gas_veldisps_xyz_array()
        return veldisps_xyz[2]**2 * ah.kms_to_cms**2 * self.get_midplane_gas_dens_array()*ah.Msol_to_g/ah.pc_to_cm**3 / ah.kB_cgs

    def get_gas_thermpress_array(self) -> np.array:
        '''Gas thermal pressure in cgs units'''
        Pth, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["masses"]*self.gas_data["U"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return Pth * (ah.gamma-1.) / ah.kB_cgs * ah.mu * ah.mp_cgs / ah.Msol_to_g / (self.resolution/ah.pc_to_cm)**3