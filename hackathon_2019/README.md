# Test Code

First fetch the data:

    %>  cd data
    %>  ./fetch.sh

Build the test program by specifying the Makefile:

    %>  make -f Makefile.gnu clean
    %>  make -f Makefile.gnu

Then set the OpenMP threads and run:

    %>  export OMP_NUM_THREADS=4
    %>  ./toast_pointing ./data/focalplane.txt \
        ./data/boresight.txt ./data/check.txt

This runs the test and compares the output to previously generated data
values.

# Choice of Problem Size

Imagine that we want to process data from a future ground-based "large
aperature telescope" which has 30000 detectors sampled at 100Hz.  A year of
data at 100% observing efficiency (or several years with a more realistic
efficiency) would require:

    30000 * (100 * 3600 * 24 * 365) * 8 bytes = 757 TB

of memory for a single copy of the timestream data.  On the cori-KNL system
we have 9600 nodes each with ~90GB of usable memory.  So the aggregate total
is 864TB.  If we ran with 16 MPI processes per node there would be 153600 total
processes.

The dataset above is naturally split into 30-50 "observations" per day.  For
this test we will assume that a single process has 37 detectors for 90
observations of 30 minutes each.  This approximates a realistic data
distribution and cost of operations like the pointing calculation.
