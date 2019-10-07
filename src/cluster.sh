#!/usr/bin/env sh
#SBATCH --partition sleuths
#SBATCH -LXserver

##SBATCH --reservation triesch-shared
##SBATCH --exclude springtalk

srun -u python3 training.py "$@"
