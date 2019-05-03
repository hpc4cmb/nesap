
#include <utils.hpp>
#include <pointing_cuda.hpp>
#include <solver_cuda.hpp>

#include <cmath>
#include <sstream>
#include <iostream>
#include <cstring>

#include <cuda_runtime.h>

#ifdef HAVE_NVTX
#include <nvToolsExt.h>
#endif

// 2/PI
#define TWOINVPI 0.63661977236758134308

// 2/3
#define TWOTHIRDS 0.66666666666666666667


static void CudaError(cudaError_t err, char const * file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(err) (CudaError(err, __FILE__, __LINE__))

// Healpix operations needed for this test.

typedef struct {
    int64_t nside;
    int64_t npix;
    int64_t ncap;
    double dnside;
    int64_t twonside;
    int64_t fournside;
    int64_t nsideplusone;
    int64_t nsideminusone;
    double halfnside;
    double tqnside;
    int64_t factor;
    int64_t jr[12];
    int64_t jp[12];
    uint64_t utab[0x100];
    uint64_t ctab[0x100];
} hpix;

void hpix_init(hpix * hp, int64_t nside) {
    hp->nside = nside;
    hp->ncap = 2 * (nside * nside - nside);
    hp->npix = 12 * nside * nside;
    hp->dnside = static_cast <double> (nside);
    hp->twonside = 2 * nside;
    hp->fournside = 4 * nside;
    hp->nsideplusone = nside + 1;
    hp->halfnside = 0.5 * (hp->dnside);
    hp->tqnside = 0.75 * (hp->dnside);
    hp->factor = 0;
    hp->nsideminusone = nside - 1;
    while (nside != (1ll << hp->factor)) {
        ++hp->factor;
    }

    static const int64_t init_jr[12] = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    memcpy(hp->jr, init_jr, sizeof(init_jr));

    static const int64_t init_jp[12] = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};
    memcpy(hp->jp, init_jp, sizeof(init_jp));

    for (uint64_t m = 0; m < 0x100; ++m) {
        hp->utab[m] = (m & 0x1) | ((m & 0x2) << 1) | ((m & 0x4) << 2) |
                   ((m & 0x8) << 3) | ((m & 0x10) << 4) | ((m & 0x20) << 5) |
                   ((m & 0x40) << 6) | ((m & 0x80) << 7);

        hp->ctab[m] = (m & 0x1) | ((m & 0x2) << 7) | ((m & 0x4) >> 1) |
                   ((m & 0x8) << 6) | ((m & 0x10) >> 2) | ((m & 0x20) << 5) |
                   ((m & 0x40) >> 3) | ((m & 0x80) << 4);
    }
    return;
}

__device__ uint64_t hpix_xy2pix(hpix * hp, uint64_t x, uint64_t y) {
    return hp->utab[x & 0xff] | (hp->utab[(x >> 8) & 0xff] << 16) |
           (hp->utab[(x >> 16) & 0xff] << 32) |
           (hp->utab[(x >> 24) & 0xff] << 48) |
           (hp->utab[y & 0xff] << 1) | (hp->utab[(y >> 8) & 0xff] << 17) |
           (hp->utab[(y >> 16) & 0xff] << 33) |
           (hp->utab[(y >> 24) & 0xff] << 49);
}

__device__ void hpix_vec2zphi(hpix * hp, double const * vec,
                              double * phi, int * region, double * z,
                              double * rtz) {
    // region encodes BOTH the sign of Z and whether its
    // absolute value is greater than 2/3.
    (*z) = vec[2];
    double za = fabs(*z);
    int itemp = ((*z) > 0.0) ? 1 : -1;
    (*region) = (za <= TWOTHIRDS) ? itemp : itemp + itemp;
    (*rtz) = sqrt(3.0 * (1.0 - za));
    (*phi) = atan2(vec[1], vec[0]);
    return;
}

__device__ void hpix_zphi2nest(hpix * hp, double phi, int region, double z,
                               double rtz, int64_t * pix) {
    double tt = (phi >= 0.0) ? phi * TWOINVPI : phi * TWOINVPI + 4.0;
    int64_t x;
    int64_t y;
    double temp1;
    double temp2;
    int64_t jp;
    int64_t jm;
    int64_t ifp;
    int64_t ifm;
    int64_t face;
    int64_t ntt;
    double tp;

    if ((region == 1) || (region == -1)) {
        temp1 = hp->halfnside + hp->dnside * tt;
        temp2 = hp->tqnside * z;

        jp = (int64_t)(temp1 - temp2);
        jm = (int64_t)(temp1 + temp2);

        ifp = jp >> hp->factor;
        ifm = jm >> hp->factor;

        if (ifp == ifm) {
            face = (ifp == 4) ? (int64_t)4 : ifp + 4;
        } else if (ifp < ifm) {
            face = ifp;
        } else {
            face = ifm + 8;
        }

        x = jm & hp->nsideminusone;
        y = hp->nsideminusone - (jp & hp->nsideminusone);
    } else {
        ntt = (int64_t)tt;

        tp = tt - (double)ntt;

        temp1 = hp->dnside * rtz;

        jp = (int64_t)(tp * temp1);
        jm = (int64_t)((1.0 - tp) * temp1);

        if (jp >= hp->nside) {
            jp = hp->nsideminusone;
        }
        if (jm >= hp->nside) {
            jm = hp->nsideminusone;
        }

        if (z >= 0) {
            face = ntt;
            x = hp->nsideminusone - jm;
            y = hp->nsideminusone - jp;
        } else {
            face = ntt + 8;
            x = jp;
            y = jm;
        }
    }

    uint64_t sipf = hpix_xy2pix(hp, (uint64_t)x, (uint64_t)y);

    (*pix) = (int64_t)sipf + (face << (2 * hp->factor));

    return;
}

__device__ void hpix_zphi2ring(hpix * hp, double phi, int region, double z,
                               double rtz, int64_t * pix) {
    double tt = (phi >= 0.0) ? phi * TWOINVPI : phi * TWOINVPI + 4.0;
    double tp;
    int64_t longpart;
    double temp1;
    double temp2;
    int64_t jp;
    int64_t jm;
    int64_t ip;
    int64_t ir;
    int64_t kshift;

    if ((region == 1) || (region == -1)) {
        temp1 = hp->halfnside + hp->dnside * tt;
        temp2 = hp->tqnside * z;

        jp = (int64_t)(temp1 - temp2);
        jm = (int64_t)(temp1 + temp2);

        ir = hp->nsideplusone + jp - jm;
        kshift = 1 - (ir & 1);

        ip = (jp + jm - hp->nside + kshift + 1) >> 1;
        ip = ip % hp->fournside;

        (*pix) = hp->ncap + ((ir - 1) * hp->fournside + ip);
    } else {
        tp = tt - floor(tt);

        temp1 = hp->dnside * rtz;

        jp = (int64_t)(tp * temp1);
        jm = (int64_t)((1.0 - tp) * temp1);
        ir = jp + jm + 1;
        ip = (int64_t)(tt * (double)ir);
        longpart = (int64_t)(ip / (4 * ir));
        ip -= longpart;

        (*pix) = (region > 0) ? (2 * ir * (ir - 1) + ip)
                 : (hp->npix - 2 * ir * (ir + 1) + ip);
    }

    return;
}

__device__ void hpix_vec2nest(hpix * hp, double const * vec, int64_t * pix) {
    double z;
    double rtz;
    double phi;
    int region;
    hpix_vec2zphi(hp, vec, &phi, &region, &z, &rtz);
    hpix_zphi2nest(hp, phi, region, z, rtz, pix);
    return;
}

__device__ void hpix_vec2ring(hpix * hp, double const * vec, int64_t * pix) {
    double z;
    double rtz;
    double phi;
    int region;
    hpix_vec2zphi(hp, vec, &phi, &region, &z, &rtz);
    hpix_zphi2ring(hp, phi, region, z, rtz, pix);
    return;
}


// Quaternion operations needed for this test

__host__ void qa_normalize_inplace(size_t n, double * q) {
    for (size_t i = 0; i < n; ++i) {
        size_t off = 4 * i;
        double norm = 0.0;
        for (size_t j = 0; j < 4; ++j) {
            norm += q[off + j] * q[off + j];
        }
        norm = 1.0 / ::sqrt(norm);
        for (size_t j = 0; j < 4; ++j) {
            q[off + j] *= norm;
        }
    }
    return;
}

__device__ void qa_rotate(double const * q_in, double const * v_in,
                          double * v_out) {
    // The input quaternion has already been normalized on the host.

    double xw =  q_in[3] * q_in[0];
    double yw =  q_in[3] * q_in[1];
    double zw =  q_in[3] * q_in[2];
    double x2 = -q_in[0] * q_in[0];
    double xy =  q_in[0] * q_in[1];
    double xz =  q_in[0] * q_in[2];
    double y2 = -q_in[1] * q_in[1];
    double yz =  q_in[1] * q_in[2];
    double z2 = -q_in[2] * q_in[2];

    v_out[0] = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] +
                            (yw + xz) * v_in[2]) + v_in[0];

    v_out[1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] +
                            (yz - xw) * v_in[2]) + v_in[1];

    v_out[2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] +
                            (x2 + y2) * v_in[2]) + v_in[2];

    return;
}

__device__ void qa_mult(double const * p, double const * q, double * r) {
    r[0] =  p[0] * q[3] + p[1] * q[2] -
               p[2] * q[1] + p[3] * q[0];
    r[1] = -p[0] * q[2] + p[1] * q[3] +
               p[2] * q[0] + p[3] * q[1];
    r[2] =  p[0] * q[1] - p[1] * q[0] +
               p[2] * q[3] + p[3] * q[2];
    r[3] = -p[0] * q[0] - p[1] * q[1] -
               p[2] * q[2] + p[3] * q[3];
    return;
}


__device__ void stokes_weights(double hwpang, double cal, double eta,
                               double const * dir, double const * orient,
                               float * weights) {
    double by = orient[0] * dir[1] - orient[1] * dir[0];
    double bx = orient[0] * (-dir[2] * dir[0]) +
         orient[1] * (-dir[2] * dir[1]) +
         orient[2] * (dir[0] * dir[0] + dir[1] * dir[1]);
    double ang = atan2(by, bx);
    ang += 2.0 * hwpang;
    ang *= 2.0;
    double sang;
    double cang;
    sincos(ang, &sang, &cang);

    weights[0] = __double2float_rn(cal);
    weights[1] = __double2float_rn(cang * eta * cal);
    weights[2] = __double2float_rn(sang * eta * cal);
    return;
}


__global__ void single_detector_nest(
        hpix * hp,
        double cal,
        double eps,
        double const * detquat,
        int nsamp,
        double const * hwpang,
        double const * boresight,
        int64_t * detpixels,
        float * detweights
    ) {
    // This is the kernel function that works on one detector for some
    // number of samples.
    double xaxis[3] = {1.0, 0.0, 0.0};
    double zaxis[3] = {0.0, 0.0, 1.0};
    double eta = (1.0 - eps) / (1.0 + eps);
    double dir[3];
    double orient[3];
    double quat[4];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nsamp;
         i += blockDim.x * gridDim.x) {
        qa_mult(&(boresight[4 * i]), detquat, quat);
        qa_rotate(quat, zaxis, dir);
        hpix_vec2nest(hp, dir, &(detpixels[i]));
        qa_rotate(quat, xaxis, orient);
        stokes_weights(hwpang[i], cal, eta, dir, orient, &(detweights[3 * i]));
    }
    return;
}


__global__ void single_detector_ring(
        hpix * hp,
        double cal,
        double eps,
        double const * detquat,
        int nsamp,
        double const * hwpang,
        double const * boresight,
        int64_t * detpixels,
        float * detweights
    ) {
    // This is the kernel function that works on one detector for some
    // number of samples.
    double xaxis[3] = {1.0, 0.0, 0.0};
    double zaxis[3] = {0.0, 0.0, 1.0};
    double eta = (1.0 - eps) / (1.0 + eps);
    double dir[3];
    double orient[3];
    double quat[4];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nsamp;
         i += blockDim.x * gridDim.x) {
        qa_mult(&(boresight[4 * i]), detquat, quat);
        qa_rotate(quat, zaxis, dir);
        hpix_vec2ring(hp, dir, &(detpixels[i]));
        qa_rotate(quat, xaxis, orient);
        stokes_weights(hwpang[i], cal, eta, dir, orient, &(detweights[3 * i]));
    }
    return;
}


__host__ void detector_pointing_healpix(
        int64_t nside, bool nest,
        toast::AlignedVector <double> const & boresight,
        toast::AlignedVector <double> const & hwpang,
        toast::AlignedVector <std::string> const & detnames,
        std::map <std::string, toast::AlignedVector <double> > const & detquat,
        std::map <std::string, double> const & detcal,
        std::map <std::string, double> const & deteps,
        int numSMs, cudaStream_t * streams, hpix * dev_hp,
        double * host_boresight, double * host_hwpang, double * host_detquat,
        double * dev_boresight, double * dev_hwpang, double * dev_detquat,
        int64_t * host_detpixels, float * host_detweights,
        int64_t * dev_detpixels, float * dev_detweights,
        std::map <std::string, toast::AlignedVector <int64_t> > & detpixels,
        std::map <std::string, toast::AlignedVector <double> > & detweights) {

    size_t nsamp = (size_t)(boresight.size() / 4);
    if (hwpang.size() != nsamp) {
        std::ostringstream o;
        o << "hwpang size not consistent with boresight.";
        throw std::runtime_error(o.str().c_str());
    }

    size_t ndet = detnames.size();
    if (detquat.size() != ndet) {
        std::ostringstream o;
        o << "number of det quaternions not consistent with number of names.";
        throw std::runtime_error(o.str().c_str());
    }
    if (detcal.size() != ndet) {
        std::ostringstream o;
        o << "number of det cal values not consistent with number of names.";
        throw std::runtime_error(o.str().c_str());
    }
    if (deteps.size() != ndet) {
        std::ostringstream o;
        o << "number of det eps vals not consistent with number of names.";
        throw std::runtime_error(o.str().c_str());
    }

    for (size_t d = 0; d < ndet; ++d) {
        if (detquat.count(detnames[d]) == 0) {
            std::ostringstream o;
            o << "no quaternion for det " << detnames[d];
            throw std::runtime_error(o.str().c_str());
        }
        if (detcal.count(detnames[d]) == 0) {
            std::ostringstream o;
            o << "no cal value for det " << detnames[d];
            throw std::runtime_error(o.str().c_str());
        }
        if (deteps.count(detnames[d]) == 0) {
            std::ostringstream o;
            o << "no epsilon value for det " << detnames[d];
            throw std::runtime_error(o.str().c_str());
        }
        if (detpixels.count(detnames[d]) == 0) {
            std::ostringstream o;
            o << "no pixel vector for det " << detnames[d];
            throw std::runtime_error(o.str().c_str());
        }
        if (detpixels.at(detnames[d]).size() != nsamp) {
            std::ostringstream o;
            o << "wrong size pixel vector for det " << detnames[d];
            throw std::runtime_error(o.str().c_str());
        }
        if (detweights.count(detnames[d]) == 0) {
            std::ostringstream o;
            o << "no weight vector for det " << detnames[d];
            throw std::runtime_error(o.str().c_str());
        }
        if (detweights.at(detnames[d]).size() != (3*nsamp)) {
            std::ostringstream o;
            o << "wrong size weight vector for det " << detnames[d];
            throw std::runtime_error(o.str().c_str());
        }
    }

    cudaEvent_t sevents[ndet];
    for (size_t d = 0; d < ndet; ++d) {
        CUDA_CHECK(
            cudaEventCreateWithFlags(&(sevents[d]), cudaEventDisableTiming));
    }


    for (size_t d = 0; d < ndet; ++d) {
        std::memcpy(&(host_detquat[d * 4]), detquat.at(detnames[d]).data(),
                    4 * sizeof(double));
    }

    std::memcpy(host_boresight, boresight.data(), 4 * nsamp * sizeof(double));

    std::memcpy(host_hwpang, hwpang.data(), nsamp * sizeof(double));

    qa_normalize_inplace(nsamp, host_boresight);

    CUDA_CHECK(cudaMemcpy(dev_boresight, host_boresight,
                          4 * nsamp * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev_hwpang, host_hwpang, nsamp * sizeof(double),
               cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev_detquat, host_detquat, ndet * 4 * sizeof(double),
               cudaMemcpyHostToDevice));

    // As a starting point, assume we run on the whole timestream for each
    // detector.  This may begin to approach the memory limits on the GPU,
    // at which point it is easy to add an outer loop here over chunks of
    // samples.

    // Threads per block
    int tpb = 256;

    // Blocks per Grid
    // int bpg = (int)((nsamp + tpb - 1) / tpb);
    int bpg = 32 * numSMs;

    for (size_t d = 0; d < ndet; ++d) {
        if (nest) {
            single_detector_nest <<<bpg, tpb, 0, streams[d]>>> (
                dev_hp,
                detcal.at(detnames[d]),
                deteps.at(detnames[d]),
                &(dev_detquat[4 * d]),
                nsamp,
                dev_hwpang,
                dev_boresight,
                &(dev_detpixels[d * nsamp]),
                &(dev_detweights[d * 3 * nsamp])
            );
        } else {
            single_detector_ring <<<bpg, tpb, 0, streams[d]>>> (
                dev_hp,
                detcal.at(detnames[d]),
                deteps.at(detnames[d]),
                &(dev_detquat[4 * d]),
                nsamp,
                dev_hwpang,
                dev_boresight,
                &(dev_detpixels[d * nsamp]),
                &(dev_detweights[d * 3 * nsamp])
            );
        }

        // memcopy results to host data structure.
        CUDA_CHECK(
            cudaMemcpyAsync(&(host_detpixels[d * nsamp]),
                            &(dev_detpixels[d * nsamp]),
                            nsamp * sizeof(int64_t),
                            cudaMemcpyDeviceToHost, streams[d]));
        CUDA_CHECK(
            cudaMemcpyAsync(&(host_detweights[d * 3 * nsamp]),
                            &(dev_detweights[d * 3 * nsamp]),
                            3 * nsamp * sizeof(float),
                            cudaMemcpyDeviceToHost, streams[d]));

        // Set event here so we can check when a stream is complete.
        CUDA_CHECK(cudaEventRecord(sevents[d], streams[d]));
    }

    // Loop over streams and process completed ones until they are all done.
    size_t nfinished = 0;
    std::vector <bool> is_done(ndet);
    for (size_t d = 0; d < ndet; ++d) {
        is_done[d] = false;
    }
    while (nfinished != ndet) {
        for (size_t d = 0; d < ndet; ++d) {
            if (! is_done[d]) {
                if (cudaEventQuery(sevents[d]) == cudaSuccess) {
                    #ifdef HAVE_NVTX
                    nvtxRangePushA("memcpy");
                    #endif
                    std::memcpy(detpixels[detnames[d]].data(),
                                &(host_detpixels[d * nsamp]),
                                nsamp * sizeof(int64_t));
                    auto & dw = detweights[detnames[d]];
                    size_t woff = d * 3 * nsamp;
                    for (size_t i = 0; i < nsamp; ++i) {
                        size_t off = 3 * i;
                        dw[off] = (double)host_detweights[woff + off];
                        dw[off + 1] = (double)host_detweights[woff + off + 1];
                        dw[off + 2] = (double)host_detweights[woff + off + 2];
                    }
                    nfinished += 1;
                    is_done[d] = true;
                    #ifdef HAVE_NVTX
                    nvtxRangePop();
                    #endif
                }
            }
        }
    }

    // Free memory

    for (size_t d = 0; d < ndet; ++d) {
        CUDA_CHECK(cudaEventDestroy(sevents[d]));
    }

    return;
}


void toast::pointing(
    int64_t nside, bool nest,
    toast::AlignedVector <double> const & boresight,
    toast::AlignedVector <double> const & hwpang,
    toast::AlignedVector <std::string> const & detnames,
    std::map <std::string, toast::AlignedVector <double> > const & detquat,
    std::map <std::string, double> const & detcal,
    std::map <std::string, double> const & deteps,
    std::map <std::string, toast::AlignedVector <int64_t> > & detpixels,
    std::map <std::string, toast::AlignedVector <double> > & detweights, size_t nobs) {

    size_t ndet = detnames.size();

    // Device query
    int ndevice;
    CUDA_CHECK(cudaGetDeviceCount(&ndevice));

    // Choose first device for now
    int dev_id = 0;
    CUDA_CHECK(cudaSetDevice(dev_id));

    // Find the number of SM's on this device
    int numSMs;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &numSMs, cudaDevAttrMultiProcessorCount, dev_id));

    // As a starting point, create one CUDA stream per detector.  Also create
    // one event per stream to indicate when the stream is done.
    cudaStream_t streams[ndet];

    for (size_t d = 0; d < ndet; ++d) {
        CUDA_CHECK(cudaStreamCreate(&(streams[d])));
    }

    // Copy common data to the GPU

    hpix * hp;
    CUDA_CHECK(cudaMallocHost(&hp, sizeof(hpix)));
    hpix_init(hp, nside);

    hpix * dev_hp;
    CUDA_CHECK(cudaMalloc(&dev_hp, sizeof(hpix)));

    CUDA_CHECK(cudaMemcpy(dev_hp, hp, sizeof(hpix),
                          cudaMemcpyHostToDevice));

    // The maximum number of samples across all observations.  For real data
    // we would also compute the maximum number of detectors.  For this test, the
    // number of detectors is always the same.

    size_t nsamp = hwpang.size();

    // Allocate input buffers on host and device to be re-used

    double * host_boresight;
    CUDA_CHECK(cudaMallocHost(&host_boresight, 4 * nsamp * sizeof(double)));

    double * host_hwpang;
    CUDA_CHECK(cudaMallocHost(&host_hwpang, nsamp * sizeof(double)));

    double * host_detquat;
    CUDA_CHECK(cudaMallocHost(&host_detquat, ndet * 4 * sizeof(double)));

    double * dev_boresight;
    CUDA_CHECK(cudaMalloc(&dev_boresight, 4 * nsamp * sizeof(double)));

    double * dev_hwpang;
    CUDA_CHECK(cudaMalloc(&dev_hwpang, nsamp * sizeof(double)));

    double * dev_detquat;
    CUDA_CHECK(cudaMalloc(&dev_detquat, ndet * 4 * sizeof(double)));

    // Allocate the output buffers for all detectors in device memory.  We
    // use floats for the Stokes weights and then convert to double before
    // returning.

    int64_t * dev_detpixels;
    CUDA_CHECK(cudaMalloc(&dev_detpixels, ndet * nsamp * sizeof(int64_t)));

    float * dev_detweights;
    CUDA_CHECK(cudaMalloc(&dev_detweights, ndet * 3 * nsamp * sizeof(float)));

    // Allocate pinned host memory for outputs

    int64_t * host_detpixels;
    CUDA_CHECK(cudaMallocHost(&host_detpixels, ndet * nsamp * sizeof(int64_t)));

    float * host_detweights;
    CUDA_CHECK(cudaMallocHost(&host_detweights,
                              ndet * 3 * nsamp * sizeof(float)));

    for (size_t ob = 0; ob < nobs; ++ob) {
        detector_pointing_healpix(nside, nest,
                                  boresight, hwpang,
                                  detnames, detquat,
                                  detcal, deteps, numSMs, streams, dev_hp,
                                  host_boresight, host_hwpang, host_detquat,
                                  dev_boresight, dev_hwpang, dev_detquat,
                                  host_detpixels, host_detweights,
                                  dev_detpixels, dev_detweights,
                                  detpixels, detweights);
    }

    // Free memory

    CUDA_CHECK(cudaFree(dev_detpixels));
    CUDA_CHECK(cudaFree(dev_detweights));

    CUDA_CHECK(cudaFree(dev_boresight));
    CUDA_CHECK(cudaFree(dev_hwpang));
    CUDA_CHECK(cudaFree(dev_detquat));
    CUDA_CHECK(cudaFree(dev_hp));

    CUDA_CHECK(cudaFreeHost(host_detpixels));
    CUDA_CHECK(cudaFreeHost(host_detweights));

    CUDA_CHECK(cudaFreeHost(host_boresight));
    CUDA_CHECK(cudaFreeHost(host_hwpang));
    CUDA_CHECK(cudaFreeHost(host_detquat));
    CUDA_CHECK(cudaFreeHost(hp));

    // Synchronize all streams and then destroy.
    for (size_t d = 0; d < ndet; ++d) {
        CUDA_CHECK(cudaStreamSynchronize(streams[d]));
        CUDA_CHECK(cudaStreamDestroy(streams[d]));
    }

    return;
}

//
// void toeplitz_multiply(
//         int fftlen, int nffts, int ncore, int nmiddle, int overlap,
//         fftw_plan & fplan, fftw_plan & rplan,
//         toast::AlignedVector <double> & fdata, toast::AlignedVector <double> & rdata,
//         toast::AlignedVector <double> const & filter,
//         toast::AlignedVector <double> & tod) {
//     // Note:  TOD buffer is replaced by output.
//     // We use "int" everywhere here since all the FFT math libraries use those.
//     // We would never take an FFT of 2^31 samples...
//
//     int nsamp = (int)tod.size();
//
//     // Clear the input buffer
//     std::fill(fdata.begin(), fdata.end(), 0.0);
//
//     std::vector <int> n_input(nffts);
//     std::vector <int> off_indata(nffts);
//     std::vector <int> off_infft(nffts);
//     std::vector <int> n_output(nffts);
//     std::vector <int> off_outdata(nffts);
//     std::vector <int> off_outfft(nffts);
//
//     int trank = 1;
//     #ifdef _OPENMP
//     trank = omp_get_thread_num();
//     #endif
//
//     if (nffts == 1) {
//         // one shot
//         n_input[0] = nsamp;
//         off_indata[0] = 0;
//         off_infft[0] = (fftlen - nsamp) >> 1;
//
//         n_output[0] = nsamp;
//         off_outdata[0] = 0;
//         off_outfft[0] = off_infft[0];
//
//         int bufoff = 0;
//
//         std::copy(&(tod[off_indata[0]]), &(tod[off_indata[0] + n_input[0]]),
//                   &(fdata[bufoff + off_infft[0]]));
//     } else {
//         // first fft
//         n_input[0] = fftlen - overlap;
//         if (n_input[0] > nsamp) {
//             n_input[0] = nsamp;
//         }
//         off_indata[0] = 0;
//         off_infft[0] = overlap;
//
//         n_output[0] = ncore;
//         off_outdata[0] = 0;
//         off_outfft[0] = overlap;
//
//         int bufoff = 0;
//
//         std::copy(&(tod[off_indata[0]]), &(tod[off_indata[0] + n_input[0]]),
//                   &(fdata[bufoff + off_infft[0]]));
//
//         // middle ffts
//
//         for (int k = 0; k < nmiddle; ++k) {
//             n_output[k + 1] = ncore;
//             off_outdata[k + 1] = (int)((nsamp - (nmiddle * ncore)) / 2) + k * ncore;
//             off_outfft[k + 1] = overlap;
//
//             n_input[k + 1] = nffts;
//             if (overlap > off_outdata[k + 1]) {
//                 off_indata[k + 1] = 0;
//             } else {
//                 off_indata[k + 1] = off_outdata[k + 1] - overlap;
//             }
//             off_infft[k + 1] = 0;
//
//             bufoff = (k + 1) * fftlen;
//             std::copy(
//                 &(tod[off_indata[k + 1]]),
//                 &(tod[off_indata[k + 1] + n_input[k + 1]]),
//                 &(fdata[bufoff + off_infft[k + 1]]));
//         }
//
//         // last fft
//         n_input[nffts - 1] = fftlen - overlap;
//         if (n_input[nffts - 1] > nsamp) {
//             n_input[nffts - 1] = nsamp;
//         }
//         off_indata[nffts - 1] = nsamp - n_input[nffts - 1];
//         off_infft[nffts - 1] = 0;
//
//         n_output[nffts - 1] = ncore;
//         off_outdata[nffts - 1] = nsamp - n_output[nffts - 1];
//         off_outfft[nffts - 1] = overlap;
//
//         bufoff = (nffts - 1) * fftlen;
//
//         std::copy(
//             &(tod[off_indata[nffts - 1]]),
//             &(tod[off_indata[nffts - 1] + n_input[nffts - 1]]),
//             &(fdata[bufoff + off_infft[nffts - 1]]));
//     }
//
//     // Forward FFTs
//
//     fftw_execute(fplan);
//
//     // Convolve with kernel
//
//     for (int k = 0; k < nffts; ++k) {
//         int bufoff = k * fftlen;
//         for (int i = 0; i < fftlen; ++i ) {
//             rdata[bufoff + i] *= filter[i];
//         }
//     }
//
//     // Reverse transform
//
//     fftw_execute(rplan);
//
//     // Copy back to TOD buffer
//
//     for (int k = 0; k < nffts; ++k) {
//         int bufoff = k * fftlen;
//         std::copy(
//             &(fdata[bufoff + off_outfft[k]]),
//             &(fdata[bufoff + off_outfft[k] + n_output[k]]),
//             &(tod[off_outdata[k]]));
//     }
//
//     return;
// }


__global__ void convert_local_pixels(
        int nsamp,
        int nsubmap,
        int nlocalsm,
        int64_t const * smlocal,
        int64_t * pixels
    ) {
    int64_t gsm;
    int64_t smpix;
    int64_t lsm;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nsamp;
         i += blockDim.x * gridDim.x) {
        gsm = pixels[i] / nsubmap;
        smpix = pixels[i] % nsubmap;
        lsm = smlocal[gsm];
        pixels[i] = (lsm * nsubmap) + smpix;
    }
    return;
}


__global__ void multiply_A(
        int nsamp,
        int nnz,
        double * tod,
        int64_t * pixels,
        float * weights,
        double * result
    ) {
    int poff;
    int toff;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nsamp;
         i += blockDim.x * gridDim.x) {
        poff = nnz * pixels[i];
        toff = nnz * i;
        tod[i] = 0.0;
        for (int j = 0; j < nnz; ++j) {
            tod[i] = weights[toff + j] * result[poff + j];
        }
    }
    return;
}


__global__ void multiply_AT(
        int nsamp,
        int nnz,
        double * tod,
        int64_t * pixels,
        float * weights,
        double * result
    ) {
    int poff;
    int toff;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nsamp;
         i += blockDim.x * gridDim.x) {
        poff = nnz * pixels[i];
        toff = nnz * i;
        for (int j = 0; j < nnz; ++j) {
            atomicAdd(
                &(result[poff + j]),
                weights[toff + j] * tod[i]);
        }
    }
    return;
}


void solver_lhs_obs(
        int64_t nside, bool nest,
        toast::AlignedVector <double> const & boresight,
        toast::AlignedVector <double> const & hwpang,
        toast::AlignedVector <double> const & filter,
        toast::AlignedVector <std::string> const & detnames,
        std::map <std::string, toast::AlignedVector <double> > const & detquat,
        std::map <std::string, double> const & detcal,
        std::map <std::string, double> const & deteps,
        int numSMs, cudaStream_t * streams, hpix * dev_hp,
        double * host_boresight, double * host_hwpang, double * host_detquat,
        double * dev_boresight, double * dev_hwpang, double * dev_detquat,
        int64_t * host_detpixels, float * host_detweights,
        int64_t * dev_detpixels, float * dev_detweights,
        int fftlen, int nffts, int ncore, int nmiddle, int overlap,
        // fftw_plan * fplans, fftw_plan * rplans,
        // std::map <int, toast::AlignedVector <double> > & tfdata,
        // std::map <int, toast::AlignedVector <double> > & trdata,
        int64_t nsubmap, int64_t nnz, int64_t nsmlocal, int64_t * smlocal,
        double * dev_tod, double * dev_input, double * dev_output
    ) {

    size_t nsamp = (size_t)(boresight.size() / 4);

    size_t ndet = detnames.size();

    for (size_t d = 0; d < ndet; ++d) {
        std::memcpy(&(host_detquat[d * 4]), detquat.at(detnames[d]).data(),
                    4 * sizeof(double));
    }

    std::memcpy(host_boresight, boresight.data(), 4 * nsamp * sizeof(double));

    std::memcpy(host_hwpang, hwpang.data(), nsamp * sizeof(double));

    qa_normalize_inplace(nsamp, host_boresight);

    CUDA_CHECK(cudaMemcpy(dev_boresight, host_boresight,
                          4 * nsamp * sizeof(double),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev_hwpang, host_hwpang, nsamp * sizeof(double),
               cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev_detquat, host_detquat, ndet * 4 * sizeof(double),
               cudaMemcpyHostToDevice));

    // As a starting point, assume we run on the whole timestream for each
    // detector.  This may begin to approach the memory limits on the GPU,
    // at which point it is easy to add an outer loop here over chunks of
    // samples.

    // Threads per block
    int tpb = 256;

    // Blocks per Grid
    // int bpg = (int)((nsamp + tpb - 1) / tpb);
    int bpg = 32 * numSMs;

    for (size_t d = 0; d < ndet; ++d) {
        if (nest) {
            single_detector_nest <<<bpg, tpb, 0, streams[d]>>> (
                dev_hp,
                detcal.at(detnames[d]),
                deteps.at(detnames[d]),
                &(dev_detquat[4 * d]),
                nsamp,
                dev_hwpang,
                dev_boresight,
                &(dev_detpixels[d * nsamp]),
                &(dev_detweights[d * 3 * nsamp])
            );
        } else {
            single_detector_ring <<<bpg, tpb, 0, streams[d]>>> (
                dev_hp,
                detcal.at(detnames[d]),
                deteps.at(detnames[d]),
                &(dev_detquat[4 * d]),
                nsamp,
                dev_hwpang,
                dev_boresight,
                &(dev_detpixels[d * nsamp]),
                &(dev_detweights[d * nnz * nsamp])
            );
        }

        // Convert global pixels to local pixels

        convert_local_pixels <<<bpg, tpb, 0, streams[d]>>> (
            nsamp, nsubmap, nsmlocal, smlocal, &(dev_detpixels[d * nsamp])
        );

        // Sample from starting map to create timestream

        multiply_A <<<bpg, tpb, 0, streams[d]>>> (
            nsamp, nnz, &(dev_tod[d * nsamp]), &(dev_detpixels[d * nsamp]),
            &(dev_detweights[d * nnz * nsamp]), dev_input);

        // Apply Toeplitz noise covariance to TOD.




        // Accumulate to result

        multiply_A <<<bpg, tpb, 0, streams[d]>>> (
            nsamp, nnz, &(dev_tod[d * nsamp]), &(dev_detpixels[d * nsamp]),
            &(dev_detweights[d * nnz * nsamp]), dev_output);

    }

    return;
}


void toast::solver_lhs(
    int64_t nside, bool nest,
    toast::AlignedVector <double> const & boresight,
    toast::AlignedVector <double> const & hwpang,
    toast::AlignedVector <double> const & filter,
    toast::AlignedVector <std::string> const & detnames,
    std::map <std::string, toast::AlignedVector <double> > const & detquat,
    std::map <std::string, double> const & detcal,
    std::map <std::string, double> const & deteps, size_t nobs,
    toast::AlignedVector <double> & result) {

    size_t nsamp = (size_t)(boresight.size() / 4);

    size_t ndet = detnames.size();

    // Device query
    int ndevice;
    CUDA_CHECK(cudaGetDeviceCount(&ndevice));

    // Choose first device for now
    int dev_id = 0;
    CUDA_CHECK(cudaSetDevice(dev_id));

    // Find the number of SM's on this device
    int numSMs;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &numSMs, cudaDevAttrMultiProcessorCount, dev_id));

    // As a starting point, create one CUDA stream per detector.  Also create
    // one event per stream to indicate when the stream is done.
    cudaStream_t streams[ndet];

    for (size_t d = 0; d < ndet; ++d) {
        CUDA_CHECK(cudaStreamCreate(&(streams[d])));
    }

    // Copy common data to the GPU

    hpix * hp;
    CUDA_CHECK(cudaMallocHost(&hp, sizeof(hpix)));
    hpix_init(hp, nside);

    hpix * dev_hp;
    CUDA_CHECK(cudaMalloc(&dev_hp, sizeof(hpix)));

    CUDA_CHECK(cudaMemcpy(dev_hp, hp, sizeof(hpix),
                          cudaMemcpyHostToDevice));

    // The maximum number of samples across all observations.  For real data
    // we would also compute the maximum number of detectors.  For this test, the
    // number of detectors is always the same.

    // Allocate input buffers on host and device to be re-used

    double * host_boresight;
    CUDA_CHECK(cudaMallocHost(&host_boresight, 4 * nsamp * sizeof(double)));

    double * host_hwpang;
    CUDA_CHECK(cudaMallocHost(&host_hwpang, nsamp * sizeof(double)));

    double * host_detquat;
    CUDA_CHECK(cudaMallocHost(&host_detquat, ndet * 4 * sizeof(double)));

    double * dev_boresight;
    CUDA_CHECK(cudaMalloc(&dev_boresight, 4 * nsamp * sizeof(double)));

    double * dev_hwpang;
    CUDA_CHECK(cudaMalloc(&dev_hwpang, nsamp * sizeof(double)));

    double * dev_detquat;
    CUDA_CHECK(cudaMalloc(&dev_detquat, ndet * 4 * sizeof(double)));

    // Allocate the output buffers for all detectors in device memory.  We
    // use floats for the Stokes weights and then convert to double before
    // returning.

    int64_t * dev_detpixels;
    CUDA_CHECK(cudaMalloc(&dev_detpixels, ndet * nsamp * sizeof(int64_t)));

    float * dev_detweights;
    CUDA_CHECK(cudaMalloc(&dev_detweights, ndet * 3 * nsamp * sizeof(float)));

    // Allocate pinned host memory for outputs

    int64_t * host_detpixels;
    CUDA_CHECK(cudaMallocHost(&host_detpixels, ndet * nsamp * sizeof(int64_t)));

    float * host_detweights;
    CUDA_CHECK(cudaMallocHost(&host_detweights,
                              ndet * 3 * nsamp * sizeof(float)));

    // First we must pass through the pointing once in order to build up the locally
    // hit pixels.

    // Use a typical NSIDE=16 value
    int64_t nsubmap = 12 * 16 * 16;

    // We have Stokes I/Q/U values.
    int64_t nnz = 3;

    std::set <int64_t> submaps;
    submaps.clear();

    std::map <std::string, toast::AlignedVector <int64_t> > detpixels;
    std::map <std::string, toast::AlignedVector <double> > detweights;

    for (auto const & dname : detnames) {
        detpixels[dname].clear();
        detweights[dname].clear();
        detpixels[dname].resize(hwpang.size());
        detweights[dname].resize(3 * hwpang.size());
    }

    for (size_t ob = 0; ob < nobs; ++ob) {
        std::cerr << "Compute locally hit pixels:  start observation " << ob << std::endl;
        detector_pointing_healpix(nside, nest,
                                  boresight, hwpang,
                                  detnames, detquat,
                                  detcal, deteps, numSMs, streams, dev_hp,
                                  host_boresight, host_hwpang, host_detquat,
                                  dev_boresight, dev_hwpang, dev_detquat,
                                  host_detpixels, host_detweights,
                                  dev_detpixels, dev_detweights,
                                  detpixels, detweights);
        for (auto const & dname : detnames) {
            toast::update_submaps(nsubmap, detpixels[dname], submaps);
        }
        std::cerr << "Compute locally hit pixels:  stop observation " << ob << std::endl;
    }

    // Free host memory that we no longer need
    detpixels.clear();
    detweights.clear();
    CUDA_CHECK(cudaFreeHost(host_detpixels));
    CUDA_CHECK(cudaFreeHost(host_detweights));

    // Now allocate the result map
    int64_t nsmlocal = submaps.size();
    std::vector <int64_t> smlocal(nsmlocal);
    std::fill(smlocal.begin(), smlocal.end(), -1);

    int64_t sm = 0;
    for (auto const & smap : submaps) {
        smlocal[smap] = sm;
        sm++;
    }
    result.resize(nnz * nsubmap * nsmlocal);
    std::fill(result.begin(), result.end(), 1.0);

    int64_t * dev_smlocal;
    CUDA_CHECK(cudaMalloc(&dev_smlocal, nsmlocal * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(dev_smlocal, smlocal.data(), nsmlocal * sizeof(int64_t),
                          cudaMemcpyHostToDevice));

    double * dev_input;
    CUDA_CHECK(cudaMalloc(&dev_input, nnz * nsubmap * nsmlocal * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dev_input, result.data(),
                          nnz * nsubmap * nsmlocal * sizeof(double),
                          cudaMemcpyHostToDevice));

    double * dev_output;
    CUDA_CHECK(cudaMalloc(&dev_output, nnz * nsubmap * nsmlocal * sizeof(double)));

    double * dev_tod;
    CUDA_CHECK(cudaMalloc(&dev_tod, nsamp * sizeof(double)));

    // NOTE:  we are "cheating" here since we know that observations are all the same
    // length.  Normally we would cache all the plan lengths and batch sizes that are
    // used throughout the code...

    int npsd = filter.size();
    int half = npsd - 1;
    int fftlen = 4 * half;

    int overlap = half;

    int nffts;
    int nmiddle = 0;
    int ncore = fftlen - 2 * overlap;

    if (nsamp < (size_t)ncore) {
        // do it in one shot
        nffts = 1;
    } else {
        // we have at least the first and last ffts
        nffts = 2;

        if (nsamp <= 2 * (size_t)ncore) {
            nmiddle = 0;
        } else {
            nmiddle = (size_t)((nsamp - 2 * (size_t)ncore) / (size_t)ncore) + 1;
        }

        nffts += nmiddle;
    }

    // Compute the full symmetric fourier space kernel

    toast::AlignedVector <double> filtkern(fftlen);

    double orig_norm = 0.0;
    for (auto const & val : filter) {
        orig_norm += val;
    }

    filtkern[0] = filter[0];
    filtkern[half * 2] = filter[half];

    double norm = filtkern[0] + filtkern[half * 2];

    for (size_t i = 1; i < 2 * half; ++i) {
        size_t lower = (size_t)(i / 2);
        if (i % 2 == 0) {
            filtkern[i] = filter[lower];
        } else {
            filtkern[i] = 0.5 * (filter[lower] + filter[lower + 1]);
        }
        norm += filtkern[i];
    }

    double scale = orig_norm / norm;
    for (size_t i = 0; i <= 2 * half; ++i) {
        filtkern[i] *= scale;
    }

    // Create the FFT plans



    // Submit the solver kernels

    for (size_t ob = 0; ob < nobs; ++ob) {
        std::cerr << "Compute solver LHS:  start observation " << ob << std::endl;
        solver_lhs_obs(
            nside, nest, boresight, hwpang, filtkern, detnames,
            detquat, detcal, deteps, numSMs, streams, dev_hp, host_boresight,
            host_hwpang, host_detquat, dev_boresight, dev_hwpang, dev_detquat,
            host_detpixels, host_detweights, dev_detpixels, dev_detweights,
            fftlen, nffts, ncore, nmiddle, overlap,
            // FFT info here...
            nsubmap, nnz, nsmlocal, dev_smlocal, dev_tod, dev_input, dev_output
        );
        std::cerr << "Compute solver LHS:  stop observation " << ob << std::endl;
    }

    // Copy result from device memory back to host.


    // Synchronize streams.

    for (size_t d = 0; d < ndet; ++d) {
        CUDA_CHECK(cudaStreamSynchronize(streams[d]));
    }

    // Free memory

    CUDA_CHECK(cudaFree(dev_smlocal));
    CUDA_CHECK(cudaFree(dev_tod));
    CUDA_CHECK(cudaFree(dev_input));
    CUDA_CHECK(cudaFree(dev_output));

    CUDA_CHECK(cudaFree(dev_detpixels));
    CUDA_CHECK(cudaFree(dev_detweights));

    CUDA_CHECK(cudaFree(dev_boresight));
    CUDA_CHECK(cudaFree(dev_hwpang));
    CUDA_CHECK(cudaFree(dev_detquat));
    CUDA_CHECK(cudaFree(dev_hp));

    CUDA_CHECK(cudaFreeHost(host_boresight));
    CUDA_CHECK(cudaFreeHost(host_hwpang));
    CUDA_CHECK(cudaFreeHost(host_detquat));
    CUDA_CHECK(cudaFreeHost(hp));

    // Destroy all streams.
    for (size_t d = 0; d < ndet; ++d) {
        CUDA_CHECK(cudaStreamDestroy(streams[d]));
    }












    return;
}
