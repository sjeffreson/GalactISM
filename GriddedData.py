from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np

import h5py
import astro_helper as ah

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
        root_dir: Path = DEFAULT_ROOT_DIR,
        snapname: str = "snap-DESPOTIC_300.hdf5",
        realign_galaxy: bool=True, # according to angular momentum vector of gas
    ):
        
        self.galaxy_type = galaxy_type
        self.max_grid_width = max_grid_width * ah.kpc_to_cm
        self.max_grid_height = max_grid_height * ah.kpc_to_cm
        self.resolution = resolution * ah.pc_to_cm
        self.root_dir = root_dir
        self.snapshot = snapshot
        self.realign_galaxy = realign_galaxy

        self.gas_data = self.read_property(snapshot, 0)
        if realign_galaxy:
            self.x_CM, self.y_CM, self.z_CM, self.vx_CM, self.vy_CM, self.vz_CM = self.get_gas_disk_COM(self.gas_data)
            self.Lx, self.Ly, self.Lz = self.get_gas_disk_angmom(self.gas_data)
            self.gas_data = self.realign_galaxy(self.gas_data)

        # (degraded) grid on which to compute arrays
        # TODO: Replace inverse distance interpolation with RBF interpolation package
        # TODO: Replace gridding in general with RBF interpolation package, right now it's just nearest neighbor
        xy_binnum = int(np.ceil(2.*self.max_grid_width/self.resolution))
        z_binnum = int(np.ceil(2.*self.max_grid_height/self.resolution))
        xbin_edges = np.linspace(-self.max_grid_width, self.max_grid_width, xy_binnum+1)
        ybin_edges = np.linspace(-self.max_grid_width, self.max_grid_width, xy_binnum+1)
        zbin_edges = np.linspace(-self.max_grid_height, self.max_grid_height, z_binnum+1)
        xbin_centres = (self.xbin_edges[1:]+self.xbin_edges[:-1])*0.5
        ybin_centres = (self.ybin_edges[1:]+self.ybin_edges[:-1])*0.5
        zbin_centres = (self.zbin_edges[1:]+self.zbin_edges[:-1])*0.5

        self.x_grid, self.y_grid, self.z_grid = np.meshgrid(xbin_centres, ybin_centres, zbin_centres)
    
    def read_property(
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
        snap_data["time"] = header.attrs["Time"]
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
            snap_data["masses"] = np.ones(len(snap_data["x_coords"])) * header.attrs['MassTable'][PartType] * PartType_data['Masses'].attrs['to_cgs']
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
    
    def cut_out_gas_disk(self, gas_data: Dict[str, np.array]) -> Dict[str, np.array]:
        '''cut out most of the gas cells that are in the background grid, not the disk'''
        cnd = (gas_data["R_coords"] < self.max_grid_width/2.) & (np.fabs(gas_data["z_coords"]) < self.max_grid_width/2.)
        return {key: value[cnd] for key, value in gas_data.items()}

    def get_gas_disk_COM(self, gas_data: Dict[str, np.array]) -> Tuple[float, float, float, float, float, float]:
        '''Get the center of mass (positional and velocity) of the gas disk'''
        gas_data_cut = self.cut_out_gas_disk(gas_data)

        x_CM = np.average(gas_data_cut["x_coords"], weights=gas_data_cut["masses"])
        y_CM = np.average(gas_data_cut["y_coords"], weights=gas_data_cut["masses"])
        z_CM = np.average(gas_data_cut["z_coords"], weights=gas_data_cut["masses"])
        vx_CM = np.average(gas_data_cut["velxs"], weights=gas_data_cut["masses"])
        vy_CM = np.average(gas_data_cut["velys"], weights=gas_data_cut["masses"])
        vz_CM = np.average(gas_data_cut["velzs"], weights=gas_data_cut["masses"])

        return x_CM, y_CM, z_CM, vx_CM, vy_CM, vz_CM

    def get_gas_disk_angmom(self, gas_data: Dict[str, np.array]) -> Tuple[float, float, float]:
        '''Get the angular momentum vector of the gas disk'''
        x_CM, y_CM, z_CM, vx_CM, vy_CM, vz_CM = self.get_gas_disk_COM(snap_data)
        gas_data_cut = self.cut_out_gas_disk(gas_data)

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

    def realign_galaxy(self, snap_data: Dict[str, np.array]) -> Dict[str, np.array]:
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
        snap_data['z_coords'] = zu[0]*x + zu[1]*y + zu[2]*z
        snap_data['velxs'] = xu[0]*vx + xu[1]*vy + xu[2]*vz
        snap_data['velys'] = yu[0]*vx + yu[1]*vy + yu[2]*vz
        snap_data['velzs'] = zu[0]*vx + zu[1]*vy + zu[2]*vz

        return snap_data

    def get_potential_grid(self, leafsize: int=10, eps: float=6., p: int=1) -> np.array:
        '''Get the potential array from the gas cells, in cgs units. Leafsize, eps and p
        are performance parameters for the inverse distance interpolation. See description
        in invdisttree.py'''

        if self.ptl_resolution == None:
            raise ValueError("The vertical resolution for computation of the potential must be set to use this function.")

        '''the z-grid for calculation of the potential needs to be sufficiently-fine to capture
        density fluctations near the simulation resolution'''
        ptl_z_binnum = int(np.ceil(2.*self.max_grid_height/self.ptl_resolution))
        ptl_z_bin_edges = np.linspace(-self.max_grid_height, self.max_grid_height, ptl_z_binnum+1)
        ptl_z_bin_centres = (ptl_z_bin_edges[1:]+ptl_z_bin_edges[:-1])*0.5

        snapdata_all = [self.read_property(i) for i in range(6) if self.read_property(i) != None]
        snapdata_all = [self.realign_galaxy(snapdata) for snapdata in snapdata_all]
        x_all = np.concatenate([snapdata["x_coords"] for snapdata in snapdata_all])
        y_all = np.concatenate([snapdata["y_coords"] for snapdata in snapdata_all])
        z_all = np.concatenate([snapdata["z_coords"] for snapdata in snapdata_all])
        ptl_all = np.concatenate([snapdata["Potential"] for snapdata in snapdata_all])

        invdisttree = Invdisttree(np.array([x_all, y_all, z_all]).T, ptl_all, leafsize=leafsize, stat=1)
        ptl_grid = invdisttree(np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()]).T, nnear=8, eps=eps, p=p)
        ptl_grid = np.reshape(ptl_grid, np.shape(x_grid))

        return ptl_grid

    def get_rho_potential_grid(self) -> np.array:
        '''Get the density array corresponding to the potential grid, for computation of
        interstellar medium weights'''

        if self.ptl_resolution == None:
            raise ValueError("The vertical resolution for computation of the potential must be set to use this function.")

        ptl_z_binnum = int(np.ceil(2.*self.max_grid_height/self.ptl_resolution))
        ptl_z_bin_edges = np.linspace(-self.max_grid_height, self.max_grid_height, ptl_z_binnum+1)
        ptl_z_bin_centres = (ptl_z_bin_edges[1:]+ptl_z_bin_edges[:-1])*0.5

        cnd = (snapdata["AlphaVir"] > 2.) | (snapdata["AlphaVir"] <= 0.) # cut out gas that's in gravitationally-bound clouds
        x = snapdata["x_coords"][cnd]
        y = snapdata["y_coords"][cnd]
        z = snapdata["z_coords"][cnd]
        mass = snapdata["masses"][cnd]

        rho3Ds = np.zeros((self.xy_binnum, self.xy_binnum, ptl_z_binnum))
        for zbmin, zbmax, k in zip(ptl_zbin_edges[:-1], ptl_zbin_edges[1:], range(ptl_zbinnum)):
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

    def get_weights(self) -> np.array:
        '''Get the weights for the interstellar medium, based on the density and potential
        grids'''
        rho_grid = self.get_rho_potential_grid()
        ptl_grid = self.get_potential_grid()
        
        return rho_grid/np.sqrt(np.fabs(ptl_grid))

    def get_gas_surfdens_grid(self) -> np.array:
        '''Gas surface density in solar masses per pc^2'''
        surfdens, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["masses"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return surfdens/ah.Msol_to_g/(self.resolution/ah.pc_to_cm)**2

    def get_stellar_surfdens_grid(self) -> np.array:
        '''Stellar surface density in solar masses per pc^2'''
        stardata_all = [self.read_property(i) for i in [2,3,4] if self.read_property(i) != None]
        stardata_all = [self.realign_galaxy(stardata) for stardata in stardata_all]
        x_stars = np.concatenate([snapdata["x_coords"] for snapdata in snapdata_all])
        y_stars = np.concatenate([snapdata["y_coords"] for snapdata in snapdata_all])
        z_stars = np.concatenate([snapdata["z_coords"] for snapdata in snapdata_all])
        mass_stars = np.concatenate([snapdata["masses"] for snapdata in snapdata_all])

        surfdens, x_edge, y_edge, binnumber = binned_statistic_2d(
            x_stars, y_stars, mass_stars,
            bins=(xbin_edges, ybin_edges),
            statistic='sum'
        )
        return surfdens/ah.Msol_to_g/(self.resolution/ah.pc_to_cm)**2

    def get_gas_SFR_surfdens_grid(self) -> np.array:
        '''Gas star formation rate surface density in solar masses per kpc^2 per yr'''
        SFR, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["SFRs"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return SFR/ah.Msol_to_g/(self.resolution/ah.kpc_to_cm)**2*ah.yr_to_s

    def get_midplane_gas_dens_grid(self) -> np.array:
        '''Gas volume density in solar masses per pc^3'''
        dens, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["masses"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return dens/ah.Msol_to_g/(self.resolution/ah.pc_to_cm)**3

    def get_midplane_stellar_dens_grid(self) -> np.array:
        '''Stellar volume density in solar masses per pc^3'''
        stardata_all = [self.read_property(i) for i in [2,3,4] if self.read_property(i) != None]
        stardata_all = [self.realign_galaxy(stardata) for stardata in stardata_all]
        x_stars = np.concatenate([snapdata["x_coords"] for snapdata in snapdata_all])
        y_stars = np.concatenate([snapdata["y_coords"] for snapdata in snapdata_all])
        z_stars = np.concatenate([snapdata["z_coords"] for snapdata in snapdata_all])
        mass_stars = np.concatenate([snapdata["masses"] for snapdata in snapdata_all])

        dens, x_edge, y_edge, binnumber = binned_statistic_2d(
            x_stars, y_stars, mass_stars,
            bins=(xbin_edges, ybin_edges),
            statistic='sum'
        )
        return dens/ah.Msol_to_g/(self.resolution/ah.pc_to_cm)**3

    def get_midplane_SFR_dens_grid(self) -> np.array:
        '''Gas star formation rate volume density in solar masses per kpc^3 per yr'''
        Sfr, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["SFRs"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return Sfr/ah.Msol_to_g/(self.resolution/ah.kpc_to_cm)**3*ah.yr_to_s

    def get_gas_veldisps_xyz(self) -> Tuple[np.array, np.array, np.array]:
        '''Gas velocity dispersion components in km/s'''
        sumrhos, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["masses"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )

        veldisps_xyz = []
        maxbinnumber = int(np.rint(2.*self.max_grid_width*ah.kpc_to_cm/self.resolution))
        for velstring in ["velxs", "velys", "velzs"]:
            summeanvel, x_edge, y_edge, binnumber = binned_statistic_2d(
                self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["masses"]*self.gas_data[velstring],
                bins=(self.xbin_edges, self.ybin_edges),
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
            veldisps_xyp.append(np.sqrt(sumveldisp/sumrhos)/ah.kms_to_cms)

        return tuple(veldisps_xyz)

    def get_gas_turbpress_grid(self) -> np.array:
        '''Gas turbulent pressure in cgs units'''
        veldisps_xyz = self.get_gas_veldisps_xyz()
        return veldisps_xyz[2]**2 * self.get_midplane_gas_dens_grid()*ah.Msol_to_g/ah.pc_to_cm**3 / ah.kB_cgs

    def get_gas_thermpress_grid(self) -> np.array:
        '''Gas thermal pressure in cgs units'''
        Pth, x_edge, y_edge, binnumber = binned_statistic_2d(
            self.gas_data["x_coords"], self.gas_data["y_coords"], self.gas_data["masses"]*self.gas_data["U"],
            bins=(self.xbin_edges, self.ybin_edges),
            statistic='sum'
        )
        return Pth * (ah.gamma-1.) / ah.kB_cgs * ah.mu * ah.mp_cgs / ah.Msol_to_g / (self.resolution/ah.pc_to_cm)**3


    def __getitem__(
        self,
        idx: int,
    ):
        """Get item at index idx
        Args:
            idx (int): index to get
        Returns:
            Tuple[Tensor, Tensor]: input and output images
        """
        input_features = [
            torch.tensor(np.load(self.files_by_feature[feature][idx]))
            for feature in self.input_features
        ]
        output_features = [
            torch.tensor(np.load(self.files_by_feature[feature][idx]))
            for feature in self.output_features
        ]
        if self.transform:
            for i, feature in enumerate(input_features):
                input_features[i] = self.transform(torch.atleast_3d(feature))
            for i, feature in enumerate(output_features):
                output_features[i] = self.transform(torch.atleast_3d(feature))
        return input_features, output_features