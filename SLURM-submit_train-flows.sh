#!/bin/bash

# target galaxy vlM
sbatch SLURM_train-flows.sh ETG-vlowmass a
sbatch SLURM_train-flows.sh ETG-vlowmass b
sbatch SLURM_train-flows.sh ETG-vlowmass c
sbatch SLURM_train-flows.sh ETG-vlowmass d

# target galaxy lowM
sbatch SLURM_train-flows.sh ETG-lowmass a
sbatch SLURM_train-flows.sh ETG-lowmass b
sbatch SLURM_train-flows.sh ETG-lowmass c
sbatch SLURM_train-flows.sh ETG-lowmass d

# target galaxy medM
sbatch SLURM_train-flows.sh ETG-medmass a
sbatch SLURM_train-flows.sh ETG-medmass b
sbatch SLURM_train-flows.sh ETG-medmass c
sbatch SLURM_train-flows.sh ETG-medmass d

# target galaxy hiM
sbatch SLURM_train-flows.sh ETG-himass a
sbatch SLURM_train-flows.sh ETG-himass b
sbatch SLURM_train-flows.sh ETG-himass c
sbatch SLURM_train-flows.sh ETG-himass d

# target galaxy NGC300
sbatch SLURM_train-flows.sh NGC300 a
sbatch SLURM_train-flows.sh NGC300 b
sbatch SLURM_train-flows.sh NGC300 c
sbatch SLURM_train-flows.sh NGC300 d

# target galaxy MW
sbatch SLURM_train-flows.sh MW a
sbatch SLURM_train-flows.sh MW b
sbatch SLURM_train-flows.sh MW c
sbatch SLURM_train-flows.sh MW d