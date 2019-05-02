#!/bin/bash

# Make sure the data files are up to date
rm -f data/*.txt
cd data
./fetch.sh
cd ..

# Setup environment
module purge
module load esslurm
module load gcc/7.3.0
module load cuda/10.0

# Launch interactive job
salloc -N 1 -C gpu -c 8 -A m1759 -t 04:00:00 --gres=gpu:1 --mem=20GB --reservation=coe_hackathon
