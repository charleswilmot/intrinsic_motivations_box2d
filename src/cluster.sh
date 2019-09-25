#!/usr/bin/env sh
#SBATCH --partition sleuths
#SBATCH --gres gpu:1
#SBATCH -LXserver
#SBATCH --mincpus 38
#SBATCH --mem=430000
##SBATCH --reservation triesch-shared

##SBATCH --exclude springtalk

srun python3 training.py --sequence-length 100 -bs 200 -ema 0.999 -nw 20 -np 1 -fe 20000 -df 0.9 --description gamma_0.9
