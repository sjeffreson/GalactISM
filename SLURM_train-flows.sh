#!/bin/bash
#SBATCH -J train-flows
#SBATCH --mail-type=NONE
#SBATCH -p sapphire # queue (partition)
#SBATCH -e ./err_log.%j
#SBATCH -o ./out_log.%j
#SBATCH -t 02:00:00 # h:min:s
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=0

cd $SLURM_SUBMIT_DIR

# load the modules
module --force purge
module load Mambaforge/22.11.1-fasrc01

# allow large core dumps
ulimit -c unlimited

# run/submit script
echo "python3 train_flow.py $1 $2"
python3 train_flow.py $1 $2