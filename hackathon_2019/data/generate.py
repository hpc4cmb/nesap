#!/usr/bin/env python3

from toast.mpi import MPI

from toast.tod.sim_tod import TODSatellite

from toast.tod import hex_layout, hex_pol_angles_qu

# Create a simple hex focalplane.  This number of detectors might be typical
# for the number per process in a full-machine run.
npix = 19
pol = hex_pol_angles_qu(npix)
fp = hex_layout(npix, 1.0, "det", "", pol)
print(fp, flush=True)

# Write this out to a simple text format
with open("focalplane.txt", "w") as f:
    for det, props in fp.items():
        q = props["quat"]
        f.write("{} {:0.15e} {:0.15e} {:0.15e} {:0.15e}\n"
                .format(det, q[0], q[1], q[2], q[3]))

# Use a typical sample rate and simulate data for one observation.  We can
# then repeat this for a typical number of observations to get the right
# data volume.
rate = 100.0
samples = 180000

tod = TODSatellite(
    MPI.COMM_WORLD,
    fp,
    samples,
    detranks=1,
    firsttime=0.0,
    rate=rate,
    spinperiod=2.0,
    spinangle=60.0,
    precperiod=5.0,
    precangle=30.0)

tod.set_prec_axis()

boresight = tod.read_boresight()
print(boresight, flush=True)

# Write this out to a simple text format
with open("boresight.txt", "w") as f:
    for q in boresight:
        f.write("{:0.15e} {:0.15e} {:0.15e} {:0.15e}\n"
                .format(q[0], q[1], q[2], q[3]))
