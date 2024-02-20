"""
    Script written by Carol Cuesta-Lazaro and Sarah Jeffreson
"""
""" Script to train normalizing flows on galaxy simulation data, in order to
    predict the star formation rate (surface density or volume density). The
    script trains flows for each galaxy type, using only training data from the
    other galaxy types, with different sets of training variables specified by the user.
"""

import sys
import pickle
import json
import numpy as np
from pathlib import Path

import astro_helper as ah
from GalDatasets import GalDataset

import torch
from torch import optim
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

DEFAULT_ROOT_DIR = Path("/n/holystore01/LABS/itc_lab/Lab/to-Carol/")
def get_data(input_features, log_features, output_features, galaxy_types, snap_frac=0.1):
    gals = GalDataset(
        input_features=input_features,
        output_features=output_features,
        galaxy_types=galaxy_types,
        snap_frac=snap_frac,
    )
    print("Number of galaxy snapshots = "+str(len(gals)))
    input_dict, outputs_dict = gals.get_data_as_1d()

    inputs = np.stack([
        np.log10(input_dict[feature]) if feature in log_features else input_dict[feature]
        for feature in input_features
    ], axis=1)
    outputs = np.stack([
        np.log10(outputs_dict[feature]) if feature in log_features else outputs_dict[feature]
        for feature in output_features
    ], axis=1)

    return inputs, outputs

def standarize_data(logX, logY):
    logX = logX[~np.isnan(logX).any(axis=1)]
    logY = logY[~np.isnan(logY).any(axis=1)]
    logX = logX[~np.isinf(logX).any(axis=1)]
    logY = logY[~np.isinf(logY).any(axis=1)]

    norm_dict = {
        'x_min': np.min(logX, axis=0),
        'y_min': np.min(logY, axis=0),
    }

    norm_dict['x_mean'] =  np.mean(logX, axis=0)
    norm_dict['x_std'] =  np.std(logX, axis=0)

    norm_dict['y_mean'] =  np.mean(logY, axis=0)
    norm_dict['y_std'] =  np.std(logY, axis=0)

    logX = (logX - norm_dict['x_mean']) / norm_dict['x_std']
    logY = (logY - norm_dict['y_mean']) / norm_dict['y_std']

    return logX, logY, norm_dict

def training_sets(set_type):
    # training features that are broadly reliable in cosmo sims
    if(set_type=='a'):
        input_features = sorted(['midplane-dens'])
    elif(set_type=='b'):
        input_features = sorted(['midplane-dens', 'weights'])
    elif(set_type=='c'):
        input_features = sorted(['midplane-dens', 'weights', 'Omegazs', 'kappa'])
    elif(set_type=='d'):
        input_features = sorted(['midplane-dens', 'weights', 'Omegazs', 'kappa', 'midplane-stellar-dens'])
    # features that are debatably reliable in cosmo sims
    elif(set_type=='e'):
        input_features = sorted(['midplane-dens', 'midplane-stellar-dens', 'weights', 'kappa',
        'midplane-Pturb', 'midplane-Pth'])
    elif(set_type=='f'):
        input_features = sorted(['midplane-dens', 'midplane-stellar-dens', 'weights', 'kappa',
        'midplane-Pturb', 'midplane-Pth', 'midplane-veldispz', 'midplane-veldisp3D'])
    else:
        print('Please choose a valid training set: a, b, c, d, e, or f')
    return input_features

def get_flow(num_features, num_context_features, num_hidden_features, num_layers=6):
    # Define the conditional normalizing flow
    base_dist = StandardNormal(shape=[1])

    transforms = []
    for _ in range(num_layers):
        transforms.append(MaskedAffineAutoregressiveTransform(
            features=num_features, 
            context_features=num_context_features,
            hidden_features=num_hidden_features,
            )
        )
        transforms.append(ReversePermutation(features=num_features))

    transform = CompositeTransform(transforms)

    flow = Flow(transform, base_dist)
    return flow

if __name__ == '__main__':
    output_path = DEFAULT_ROOT_DIR / "flows"
    target_galaxy = sys.argv[1]
    galaxy_types = ['ETG-vlowmass', 'ETG-lowmass', 'ETG-medmass', 'ETG-himass', 'NGC300', 'MW']
    galaxy_types.remove(target_galaxy)
    galaxy_types = sorted(galaxy_types)

    log_features = ['midplane-SFR-dens', 'midplane-stellar-dens', 'midplane-dens', 'weights', 'Omegazs', 'kappas',
        'midplane-Pturb', 'midplane-Pth', 'midplane-veldispz', 'midplane-veldisp3D']
    input_features = training_sets(sys.argv[2])

    output_path = output_path / target_galaxy
    output_path = output_path / ("set_"+set_type)
    output_path.mkdir(parents=True, exist_ok=True)

    num_hidden_features = 128
    num_epochs = 1000 
    batch_size = 256 
    num_layers = 6

    X, Y = get_data(
        input_features=input_features,
        log_features=log_features,
        output_features=['midplane-SFR-dens'],
        galaxy_types=galaxy_types,
        snap_frac=0.2,
    )
    X, Y, norm_dict = standarize_data(X, Y)

    # store the flow
    norm_dict = {k: v.tolist() for k, v in norm_dict.items()}
    with open(output_path / 'norm_dict.json', 'w') as f:
        json.dump(norm_dict, f)

    print('X shape = ', X.shape)
    print('Y shape = ', Y.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device = ', device)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    flow = get_flow(
        num_features = Y.shape[-1],
        num_context_features = X.shape[-1],
        num_hidden_features=num_hidden_features,
        num_layers=num_layers,
        
    )
    flow = flow.to(device)

    # Train the flow
    optimizer = optim.Adam(flow.parameters(), lr=1e-4)
    for epoch in range(num_epochs):
        permutation = torch.randperm(X.size(0))
        X = X[permutation]
        Y = Y[permutation]
        for batch_start in range(0, X.size(0), batch_size):
            batch_end = batch_start + batch_size
            x_batch = X[batch_start:batch_end]
            y_batch = Y[batch_start:batch_end]

            optimizer.zero_grad()
            log_prob = flow.log_prob(inputs=y_batch, context=x_batch)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
        if epoch % 100 == 0:
            loss_value = loss.item()
            flow = flow.to('cpu')
            with open(output_path / f'flow_e{epoch}_{loss_value:.4f}.pkl', 'wb') as f:
                pickle.dump(flow, f)
            flow = flow.to(device)

        loss_value = loss.item()
        flow_cpu = flow.to('cpu')
        with open(output_path / f'flow_e{epoch}_{loss_value:.4f}.pkl', 'wb') as f:
            pickle.dump(flow_cpu, f)
