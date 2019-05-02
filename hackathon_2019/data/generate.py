#!/usr/bin/env python3

import numpy as np

from toast.mpi import MPI

from toast.tod.sim_tod import TODSatellite

from toast.tod.sim_noise import AnalyticNoise

from toast.tod import hex_layout, hex_pol_angles_qu

# Create a simple hex focalplane.  This number of detectors might be typical
# for the number per process in a full-machine run.
npix = 37
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

# Construct a Fourier domain kernel for the noise filter

nse = AnalyticNoise(
    detectors=["fake"],
    rate={"fake": rate},
    fmin={"fake": 1.0e-5},
    fknee={"fake": 0.100},
    alpha={"fake": 1.5},
    NET={"fake": 1.0}
)

rawfreq = nse.freq("fake")
rawpsd = nse.psd("fake")

# Perform a logarithmic interpolation.  In order to avoid zero
# values, we shift the PSD by a fixed amount in frequency and
# amplitude.

psdlen = len(rawfreq)

nfft = 16384
npsd = nfft // 2 + 1
norm = rate * float(npsd - 1)
increment = rate / (nfft - 1)

psdmin = 1.0e30
for rp in rawpsd:
    if rp < psdmin:
        psdmin = rp

nyquist = rate / 2

psdshift = 0.01 * psdmin
freqshift = increment

logfreq = np.log10(rawfreq + freqshift)
logpsd = np.log10(np.sqrt(norm * rawpsd) + psdshift)

interppsd = np.log10(increment * np.arange(npsd, dtype=np.float64) + freqshift)
stepinv = np.zeros_like(rawfreq)

for b in range(psdlen - 1):
    stepinv[b] = 1.0 / (logfreq[b + 1] - logfreq[b])

ib = 0
for b in range(npsd):
    while ((ib < psdlen - 2) and (logfreq[ib + 1] < interppsd[b])):
        ib += 1
    r = (interppsd[b] - logfreq[ib]) * stepinv[ib]
    interppsd[b] = logpsd[ib] + r * (logpsd[ib + 1] - logpsd[ib])

interppsd = np.power(10, interppsd)
interppsd -= psdshift
interppsd[0] = 0.0

# Write this out to a simple text format
with open("filter.txt", "w") as f:
    for x in interppsd:
        f.write("{:0.15e}\n".format(x))
