# Test Code

First fetch the data::

    %>  cd data
    %>  ./fetch.sh

Build the test program by specifying the Makefile::

    %>  make -f Makefile.gnu

Then set the OpenMP threads and run::

    %>  export OMP_NUM_THREADS=4
    %>  ./toast_pointing ./data/focalplane.txt ./data/boresight.txt
