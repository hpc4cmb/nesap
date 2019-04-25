#!/bin/bash

# Build executables
make -f Makefile.cuda-70 clean
make -f Makefile.cuda-70

# Run the tests
export OMP_NUM_THREADS=4
echo ""
echo "RUNNING OpenMP test"
echo ""
com="srun -n 1 ./toast_pointing_omp ./data/focalplane.txt ./data/boresight.txt ./data/check.txt"
echo ${com}
eval ${com}

echo ""
echo "RUNNING CUDA test"
echo ""
com="srun -n 1 ./toast_pointing_cuda ./data/focalplane.txt ./data/boresight.txt ./data/check.txt"
echo ${com}
eval ${com}

