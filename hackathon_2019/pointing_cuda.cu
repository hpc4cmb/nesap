
#include <utils.hpp>
#include <pointing_cuda.hpp>

#include <cmath>
#include <sstream>
#include <iostream>


#include <cuda_runtime.h>

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

__host__ void hpix_init(hpix * hp, int64_t nside) {
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

    hp->jr = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    hp->jp = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};

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

//
// void toast::HealpixPixels::vec2zphi(int64_t n, double const * vec,
//                                     double * phi, int * region, double * z,
//                                     double * rtz) const {
//     if (n > std::numeric_limits <int>::max()) {
//         std::string msg("healpix vector conversion must be in chunks of < 2^31");
//         throw std::runtime_error(msg.c_str());
//     }
//
//     toast::AlignedVector <double> work1(n);
//     toast::AlignedVector <double> work2(n);
//     toast::AlignedVector <double> work3(n);
//
//     if (toast::is_aligned(vec) && toast::is_aligned(phi) &&
//         toast::is_aligned(region) && toast::is_aligned(z)
//         && toast::is_aligned(rtz)) {
//         #pragma omp simd
//         for (int64_t i = 0; i < n; ++i) {
//             int64_t offset = 3 * i;
//
//             // region encodes BOTH the sign of Z and whether its
//             // absolute value is greater than 2/3.
//
//             z[i] = vec[offset + 2];
//
//             double za = ::fabs(z[i]);
//
//             int itemp = (z[i] > 0.0) ? 1 : -1;
//
//             region[i] = (za <= TWOTHIRDS) ? itemp : itemp + itemp;
//
//             work1[i] = 3.0 * (1.0 - za);
//             work3[i] = vec[offset + 1];
//             work2[i] = vec[offset];
//         }
//     } else {
//         for (int64_t i = 0; i < n; ++i) {
//             int64_t offset = 3 * i;
//
//             // region encodes BOTH the sign of Z and whether its
//             // absolute value is greater than 2/3.
//
//             z[i] = vec[offset + 2];
//
//             double za = ::fabs(z[i]);
//
//             int itemp = (z[i] > 0.0) ? 1 : -1;
//
//             region[i] = (za <= TWOTHIRDS) ? itemp : itemp + itemp;
//
//             work1[i] = 3.0 * (1.0 - za);
//             work3[i] = vec[offset + 1];
//             work2[i] = vec[offset];
//         }
//     }
//
//     toast::vfast_sqrt(n, work1.data(), rtz);
//     toast::vfast_atan2(n, work3.data(), work2.data(), phi);
//
//     return;
// }
//
// void toast::HealpixPixels::zphi2nest(int64_t n, double const * phi,
//                                      int const * region, double const * z,
//                                      double const * rtz, int64_t * pix) const {
//     if (n > std::numeric_limits <int>::max()) {
//         std::string msg("healpix vector conversion must be in chunks of < 2^31");
//         throw std::runtime_error(msg.c_str());
//     }
//     if (toast::is_aligned(phi) && toast::is_aligned(pix) &&
//         toast::is_aligned(region) && toast::is_aligned(z)
//         && toast::is_aligned(rtz)) {
//         #pragma omp simd
//         for (int64_t i = 0; i < n; ++i) {
//             double tt =
//                 (phi[i] >= 0.0) ? phi[i] * TWOINVPI : phi[i] * TWOINVPI + 4.0;
//
//             int64_t x;
//             int64_t y;
//             double temp1;
//             double temp2;
//             int64_t jp;
//             int64_t jm;
//             int64_t ifp;
//             int64_t ifm;
//             int64_t face;
//             int64_t ntt;
//             double tp;
//
//             if (::abs(region[i]) == 1) {
//                 temp1 = halfnside_ + dnside_ * tt;
//                 temp2 = tqnside_ * z[i];
//
//                 jp = static_cast <int64_t> (temp1 - temp2);
//                 jm = static_cast <int64_t> (temp1 + temp2);
//
//                 ifp = jp >> factor_;
//                 ifm = jm >> factor_;
//
//                 face;
//                 if (ifp == ifm) {
//                     face = (ifp == 4) ? static_cast <int64_t> (4) : ifp + 4;
//                 } else if (ifp < ifm) {
//                     face = ifp;
//                 } else {
//                     face = ifm + 8;
//                 }
//
//                 x = jm & nsideminusone_;
//                 y = nsideminusone_ - (jp & nsideminusone_);
//             } else {
//                 ntt = static_cast <int64_t> (tt);
//
//                 tp = tt - static_cast <double> (ntt);
//
//                 temp1 = dnside_ * rtz[i];
//
//                 jp = static_cast <int64_t> (tp * temp1);
//                 jm = static_cast <int64_t> ((1.0 - tp) * temp1);
//
//                 if (jp >= nside_) {
//                     jp = nsideminusone_;
//                 }
//                 if (jm >= nside_) {
//                     jm = nsideminusone_;
//                 }
//
//                 if (z[i] >= 0) {
//                     face = ntt;
//                     x = nsideminusone_ - jm;
//                     y = nsideminusone_ - jp;
//                 } else {
//                     face = ntt + 8;
//                     x = jp;
//                     y = jm;
//                 }
//             }
//
//             uint64_t sipf = xy2pix_(static_cast <uint64_t> (x),
//                                     static_cast <uint64_t> (y));
//
//             pix[i] = static_cast <int64_t> (sipf) + (face << (2 * factor_));
//         }
//     } else {
//         for (int64_t i = 0; i < n; ++i) {
//             double tt =
//                 (phi[i] >= 0.0) ? phi[i] * TWOINVPI : phi[i] * TWOINVPI + 4.0;
//
//             int64_t x;
//             int64_t y;
//             double temp1;
//             double temp2;
//             int64_t jp;
//             int64_t jm;
//             int64_t ifp;
//             int64_t ifm;
//             int64_t face;
//             int64_t ntt;
//             double tp;
//
//             if (::abs(region[i]) == 1) {
//                 temp1 = halfnside_ + dnside_ * tt;
//                 temp2 = tqnside_ * z[i];
//
//                 jp = static_cast <int64_t> (temp1 - temp2);
//                 jm = static_cast <int64_t> (temp1 + temp2);
//
//                 ifp = jp >> factor_;
//                 ifm = jm >> factor_;
//
//                 face;
//                 if (ifp == ifm) {
//                     face = (ifp == 4) ? static_cast <int64_t> (4) : ifp + 4;
//                 } else if (ifp < ifm) {
//                     face = ifp;
//                 } else {
//                     face = ifm + 8;
//                 }
//
//                 x = jm & nsideminusone_;
//                 y = nsideminusone_ - (jp & nsideminusone_);
//             } else {
//                 ntt = static_cast <int64_t> (tt);
//
//                 tp = tt - static_cast <double> (ntt);
//
//                 temp1 = dnside_ * rtz[i];
//
//                 jp = static_cast <int64_t> (tp * temp1);
//                 jm = static_cast <int64_t> ((1.0 - tp) * temp1);
//
//                 if (jp >= nside_) {
//                     jp = nsideminusone_;
//                 }
//                 if (jm >= nside_) {
//                     jm = nsideminusone_;
//                 }
//
//                 if (z[i] >= 0) {
//                     face = ntt;
//                     x = nsideminusone_ - jm;
//                     y = nsideminusone_ - jp;
//                 } else {
//                     face = ntt + 8;
//                     x = jp;
//                     y = jm;
//                 }
//             }
//
//             uint64_t sipf = xy2pix_(static_cast <uint64_t> (x),
//                                     static_cast <uint64_t> (y));
//
//             pix[i] = static_cast <int64_t> (sipf) + (face << (2 * factor_));
//         }
//     }
//
//     return;
// }
//
// void toast::HealpixPixels::zphi2ring(int64_t n, double const * phi,
//                                      int const * region, double const * z,
//                                      double const * rtz, int64_t * pix) const {
//     if (n > std::numeric_limits <int>::max()) {
//         std::string msg("healpix vector conversion must be in chunks of < 2^31");
//         throw std::runtime_error(msg.c_str());
//     }
//     if (toast::is_aligned(phi) && toast::is_aligned(pix) &&
//         toast::is_aligned(region) && toast::is_aligned(z)
//         && toast::is_aligned(rtz)) {
//         #pragma omp simd
//         for (int64_t i = 0; i < n; ++i) {
//             double tt =
//                 (phi[i] >= 0.0) ? phi[i] * TWOINVPI : phi[i] * TWOINVPI + 4.0;
//
//             double tp;
//             int64_t longpart;
//             double temp1;
//             double temp2;
//             int64_t jp;
//             int64_t jm;
//             int64_t ip;
//             int64_t ir;
//             int64_t kshift;
//
//             if (::abs(region[i]) == 1) {
//                 temp1 = halfnside_ + dnside_ * tt;
//                 temp2 = tqnside_ * z[i];
//
//                 jp = static_cast <int64_t> (temp1 - temp2);
//                 jm = static_cast <int64_t> (temp1 + temp2);
//
//                 ir = nsideplusone_ + jp - jm;
//                 kshift = 1 - (ir & 1);
//
//                 ip = (jp + jm - nside_ + kshift + 1) >> 1;
//                 ip = ip % fournside_;
//
//                 pix[i] = ncap_ + ((ir - 1) * fournside_ + ip);
//             } else {
//                 tp = tt - floor(tt);
//
//                 temp1 = dnside_ * rtz[i];
//
//                 jp = static_cast <int64_t> (tp * temp1);
//                 jm = static_cast <int64_t> ((1.0 - tp) * temp1);
//                 ir = jp + jm + 1;
//                 ip = static_cast <int64_t> (tt * (double)ir);
//                 longpart = static_cast <int64_t> (ip / (4 * ir));
//                 ip -= longpart;
//
//                 pix[i] = (region[i] > 0) ? (2 * ir * (ir - 1) + ip)
//                          : (npix_ - 2 * ir * (ir + 1) + ip);
//             }
//         }
//     } else {
//         for (int64_t i = 0; i < n; ++i) {
//             double tt =
//                 (phi[i] >= 0.0) ? phi[i] * TWOINVPI : phi[i] * TWOINVPI + 4.0;
//
//             double tp;
//             int64_t longpart;
//             double temp1;
//             double temp2;
//             int64_t jp;
//             int64_t jm;
//             int64_t ip;
//             int64_t ir;
//             int64_t kshift;
//
//             if (::abs(region[i]) == 1) {
//                 temp1 = halfnside_ + dnside_ * tt;
//                 temp2 = tqnside_ * z[i];
//
//                 jp = static_cast <int64_t> (temp1 - temp2);
//                 jm = static_cast <int64_t> (temp1 + temp2);
//
//                 ir = nsideplusone_ + jp - jm;
//                 kshift = 1 - (ir & 1);
//
//                 ip = (jp + jm - nside_ + kshift + 1) >> 1;
//                 ip = ip % fournside_;
//
//                 pix[i] = ncap_ + ((ir - 1) * fournside_ + ip);
//             } else {
//                 tp = tt - floor(tt);
//
//                 temp1 = dnside_ * rtz[i];
//
//                 jp = static_cast <int64_t> (tp * temp1);
//                 jm = static_cast <int64_t> ((1.0 - tp) * temp1);
//                 ir = jp + jm + 1;
//                 ip = static_cast <int64_t> (tt * (double)ir);
//                 longpart = static_cast <int64_t> (ip / (4 * ir));
//                 ip -= longpart;
//
//                 pix[i] = (region[i] > 0) ? (2 * ir * (ir - 1) + ip)
//                          : (npix_ - 2 * ir * (ir + 1) + ip);
//             }
//         }
//     }
//
//     return;
// }
//
// void toast::HealpixPixels::vec2nest(int64_t n, double const * vec,
//                                     int64_t * pix) const {
//     if (n > std::numeric_limits <int>::max()) {
//         std::string msg("healpix vector conversion must be in chunks of < 2^31");
//         throw std::runtime_error(msg.c_str());
//     }
//
//     toast::AlignedVector <double> z(n);
//     toast::AlignedVector <double> rtz(n);
//     toast::AlignedVector <double> phi(n);
//     toast::AlignedVector <int> region(n);
//
//     vec2zphi(n, vec, phi.data(), region.data(), z.data(), rtz.data());
//
//     zphi2nest(n, phi.data(), region.data(), z.data(), rtz.data(), pix);
//
//     return;
// }
//
// void toast::HealpixPixels::vec2ring(int64_t n, double const * vec,
//                                     int64_t * pix) const {
//     if (n > std::numeric_limits <int>::max()) {
//         std::string msg("healpix vector conversion must be in chunks of < 2^31");
//         throw std::runtime_error(msg.c_str());
//     }
//
//     toast::AlignedVector <double> z(n);
//     toast::AlignedVector <double> rtz(n);
//     toast::AlignedVector <double> phi(n);
//     toast::AlignedVector <int> region(n);
//
//     vec2zphi(n, vec, phi.data(), region.data(), z.data(), rtz.data());
//
//     zphi2ring(n, phi.data(), region.data(), z.data(), rtz.data(), pix);
//
//     return;
// }


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
                               double * weights) {
    double bx;
    double by;
    by = orient[0] * dir[1] - orient[1] * dir[0];
    bx = orient[0] * (-dir[2] * dir[0]) +
         orient[1] * (-dir[2] * dir[1]) +
         orient[2] * (dir[0] * dir[0] + dir[1] * dir[1]);
    double ang = atan2(by, bx)
    ang += 2.0 * hwpang;
    ang *= 2.0;
    sincos(ang, sang, cang);

    weights[0] = cal;
    weights[1] = cang * eta * cal;
    weights[2] = sang * eta * cal;
    return;
}


__global__ void single_detector_nest(
        int64_t nside,
        double cal,
        double eps,
        double const * detquat,
        size_t nsamp,
        double const * hwpang,
        double const * boresight,
        int64_t * detpixels,
        double * detweights
    ) {
    // This is the kernel function that works on one detector for some
    // number of samples.
    double xaxis[3] = {1.0, 0.0, 0.0};
    double zaxis[3] = {0.0, 0.0, 1.0};
    double eta = (1.0 - eps) / (1.0 + eps);
    double dir[3];
    double orient[3];
    double quat[4];

    for (size_t i = 0; i < nsamp; ++i) {
        qa_mult(&(boresight[4 * i]), detquat, quat);
        qa_rotate(quat, zaxis, dir);
        hpix_vec2nest(dir, &(detpixels[i]));
        qa_rotate(quat, xaxis, orient);
        stokes_weights(hwpang[i], cal, eta, dir, orient,
                       &(detweights[3 * i]));
    }
    return;
}


__global__ void single_detector_ring(
        int64_t nside,
        double cal,
        double eps,
        double const * detquat,
        size_t nsamp,
        double const * hwpang,
        double const * boresight,
        int64_t * detpixels,
        double * detweights
    ) {
    // This is the kernel function that works on one detector for some
    // number of samples.
    double xaxis[3] = {1.0, 0.0, 0.0};
    double zaxis[3] = {0.0, 0.0, 1.0};
    double eta = (1.0 - eps) / (1.0 + eps);
    double dir[3];
    double orient[3];
    double quat[4];

    for (size_t i = 0; i < nsamp; ++i) {
        qa_mult(&(boresight[4 * i]), detquat, quat);
        qa_rotate(quat, zaxis, dir);
        hpix_vec2ring(dir, &(detpixels[i]));
        qa_rotate(quat, xaxis, orient);
        stokes_weights(hwpang[i], cal, eta, dir, orient,
                       &(detweights[3 * i]));
    }
    return;
}


void toast::detector_pointing_healpix(
        int64_t nside, bool nest,
        toast::AlignedVector <double> const & boresight,
        toast::AlignedVector <double> const & hwpang,
        toast::AlignedVector <std::string> const & detnames,
        std::map <std::string, toast::AlignedVector <double> > const & detquat,
        std::map <std::string, double> const & detcal,
        std::map <std::string, double> const & deteps,
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

    // Device query

    int ndevice;
    CUDA_CHECK(cudaGetDeviceCount(&ndevice));

    std::cout << "Found " << ndevice << " CUDA devices" << std::endl;

    // Now decide how many streams we want to create- query device to find
    // number of supported streams and we should add an argument so that
    // the calling code can pass in the number of processes sharing a device.

    // As a starting point, assume we run on the whole timestream for each
    // detector.

    // Normalize boresight quaternions and copy to device along with the HWP
    // angles.

    for (size_t d = 0; d < ndet; ++d) {
        // Allocate per-detector memory for outputs
        if (nest) {
            single_detector_nest<<< >>>(
                nside, detcal[d], deteps[d], dev_detquat,
                nsamp, dev_hwpang, dev_boresight,
                dev_detpixels, dev_detweights
            );
        } else {
            single_detector_ring<<< >>>(
                nside, detcal[d], deteps[d], dev_detquat,
                nsamp, dev_hwpang, dev_boresight,
                dev_detpixels, dev_detweights
            );
        }
        // memcopy results to host data structure.
    }

    return;
}
