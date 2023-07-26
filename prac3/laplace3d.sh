#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100:1

# set name of job
#SBATCH --job-name=laplace3d

# use our reservation
#SBATCH --reservation=gputraining202307

module purge
module load CUDA

./laplace3d
