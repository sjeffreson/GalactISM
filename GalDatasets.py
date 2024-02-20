"""
    Class written by Sarah Jeffreson and Carol Cuesta-Lazaro
"""
""" Class to load dataset from galaxy simulations and perform various
    formatting operations
"""

from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
import torch

import astro_helper as ah

DEFAULT_ROOT_DIR = Path("/n/holystore01/LABS/itc_lab/Lab/to-Carol/")
class GalDataset:
    """Dataset for general galaxy simulation snapshots of a general size"""
    def __init__(
        self,
        input_features: List[str],
        output_features: List[str],
        galaxy_types: List[str],
        snap_frac: float = 0.1,
        root_dir: Path = DEFAULT_ROOT_DIR
    ):
        self.input_features = input_features
        self.output_features = output_features
        self.galaxy_types = galaxy_types
        self.snap_frac = snap_frac
        self.root_dir = root_dir

        self.all_filenames_by_feature = self.get_all_filenames_by_feature()
        self.min_num_usable_px = self.get_min_num_usable_px()

    def read_property(
        self,
        snapshot: int,
        galaxy_type: str,
        feature: str
    ) -> np.array:
        """Read a property from a snapshot as a 2D array of input size,
        including all data (all pixels) from that snapshot
        Args:
            snapshot (int): snapshot number
            galaxy_type (str): type of galaxy
            feature (str): feature to read
        Returns:
            np.array: image with feature
        """
        filename = f"{galaxy_type}/{feature}_{snapshot}.npy"
        return np.load(self.root_dir / filename)
    
    def get_idcs_usable_px(
        self,
        galaxy_type: str
    ) -> np.array:
        """Get the indices of the pixels in a galaxy that are above
        the threshold for star formation
        Args:
            galaxy_type (str): type of galaxy
        Returns:
            np.array: indices of pixels that are not zero
        """
        density = np.array(ah.flatten_list([
            list(np.ravel(np.load(item))) for item in self.get_filenames(
                galaxy_type=galaxy_type,
                feature="midplane-SFR-dens")
        ]))
        return np.where(density>0)[0]

    def get_num_usable_px(
        self,
        galaxy_type: str
    ) -> int:
        """Get the number of pixels in a galaxy that are not zero
        Args:
            galaxy_type (str): type of galaxy
        Returns:
            int: number of pixels that are not zero
        """
        idcs_usable_px = self.get_idcs_usable_px(galaxy_type)
        return len(idcs_usable_px)

    def get_min_num_usable_px(
        self,
    ) -> int:
        """Get the minimum number of pixels across all galaxies
        Returns:
            int: minimum number of pixels across all galaxies
        """
        min_num_px = np.inf
        for galaxy_type in self.galaxy_types:
            num_px = self.get_num_usable_px(galaxy_type)
            if num_px < min_num_px:
                min_num_px = num_px
        return min_num_px

    def __getitem__(
        self,
        idx: int
    ):
        """Get full (all pixels) 2D array at index idx, across all galaxy types,
        by feature.
        Args:
            idx (int): index to get
        Returns:
            Tuple[Tensor, Tensor]: input and output images
        """
        input_features = [
            torch.tensor(np.load(self.all_filenames_by_feature[feature][idx]))
            for feature in self.input_features
        ]
        output_features = [
            torch.tensor(np.load(self.all_filenames_by_feature[feature][idx]))
            for feature in self.output_features
        ]
        return input_features, output_features
    
    def __len__(
        self,
    ) -> int:
        return len(self.all_filenames_by_feature[self.input_features[0]])
    
    def get_filenames(
        self,
        galaxy_type: str,
        feature: str
    ) -> List[str]:
        """filenames in data folder
        Returns:
            List[str]: list of filenames in data folder
        """
        filename = f"{galaxy_type}/{feature}*_???.npy"
        filename_list = list(sorted(self.root_dir.rglob(filename)))
        return filename_list[::int(np.rint(1./self.snap_frac))]

    def get_all_filenames(
        self,
        feature: str,
    ):
        filenames = []
        for galaxy_type in self.galaxy_types:
            filenames += self.get_filenames(
                galaxy_type=galaxy_type, feature=feature,
            )
        return filenames

    def get_filenames_density(
        self,
        galaxy_type: str,
    ) -> List[str]:
        """filenames for the mid-plane SFR density in data folder
        Returns:
            List[str]: list of filenames in data folder
        """
        filename = f"{galaxy_type}/midplane-SFR-dens*.npy"
        filename_list = list(sorted(self.root_dir.rglob(filename)))
        return filename_list[::int(np.rint(1./self.snap_frac))]

    def get_all_filenames_density(
        self,
    ) -> List[str]:
        filenames = []
        for galaxy_type in self.galaxy_types:
            filenames += self.get_filenames_density(galaxy_type)
        return filenames

    def get_all_filenames_by_feature(
        self,
    ):
        all_filenames_by_feature = {}
        for feature in self.input_features:
            all_filenames_by_feature[feature] = self.get_all_filenames(
                feature=feature
            )
        for feature in self.output_features:
            all_filenames_by_feature[feature] = self.get_all_filenames(
                feature=feature
            )
        return all_filenames_by_feature

    def get_filenames_by_feature(
        self,
        galaxy_type: str,
    ) -> Dict[str, List[str]]:
        filenames_by_feature = {}
        for feature in self.input_features:
            filenames_by_feature[feature] = self.get_filenames(
                galaxy_type=galaxy_type,
                feature=feature
            )
        for feature in self.output_features:
            filenames_by_feature[feature] = self.get_filenames(
                galaxy_type=galaxy_type,
                feature=feature
            )
        return filenames_by_feature

    def get_data_as_1d(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get data for a feature as a 1D array, for a particular
        galaxy type. Sample only the number of pixels if specified,
        generally equal to the smallest number of pixels of any
        galaxy in the sample (so that training samples are of equal
        size). Assumes same amount of data for each feature.
        Args:
            feature: feature to analyze
            galaxy_type: self-explanatory
            num_px: number of pixels to sample (across all snapshots)
        Returns:
            Tuple[Tensor, Tensor]: input and output arrays
        """

        input_features = {feature: [] for feature in self.input_features}
        output_features = {feature: [] for feature in self.output_features}
        
        px_idcs_gals = []
        for galaxy_type in self.galaxy_types:
            nonzero_density_idcs = self.get_idcs_usable_px(galaxy_type)
            px_idcs = np.random.choice(
                nonzero_density_idcs,
                size=self.min_num_usable_px,
                replace=False
            )

            filenames_by_feature = self.get_filenames_by_feature(galaxy_type)
            for feature in self.input_features:
                input_features[feature].append(np.array(ah.flatten_list([
                    list(np.ravel(np.load(filenames_by_feature[feature][idx])))
                    for idx in range(len(filenames_by_feature[feature]))
                ]))[px_idcs])

            for feature in self.output_features:
                output_features[feature].append(np.array(ah.flatten_list([
                    list(np.ravel(np.load(filenames_by_feature[feature][idx])))
                    for idx in range(len(filenames_by_feature[feature]))
                ]))[px_idcs])

        for feature in self.input_features:
            input_features[feature] = torch.tensor(np.array(input_features[feature]).flatten())
        for feature in self.output_features:
            output_features[feature] = torch.tensor(np.array(output_features[feature]).flatten())
        
        return input_features, output_features