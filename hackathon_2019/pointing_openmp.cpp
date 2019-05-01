
#include <utils.hpp>
#include <pointing_openmp.hpp>

#ifdef HAVE_MKL
# include <mkl.h>
#endif // ifdef HAVE_MKL

#include <cmath>
#include <sstream>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif


// Vector math needed for this test.

void toast::vsqrt(int n, double const * in, double * out) {
    if (toast::is_aligned(in) && toast::is_aligned(out)) {
        # pragma omp simd
        for (int i = 0; i < n; ++i) {
            out[i] = ::sqrt(in[i]);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            out[i] = ::sqrt(in[i]);
        }
    }
    return;
}

#ifdef HAVE_MKL

// These call MKL VM functions with "Low Accuracy" mode.

void toast::vfast_sincos(int n, double const * ang, double * sinout,
                         double * cosout) {
    vmdSinCos(n, ang, sinout, cosout,
              VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vfast_atan2(int n, double const * y, double const * x,
                        double * ang) {
    vmdAtan2(n, y, x, ang, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

void toast::vfast_sqrt(int n, double const * in, double * out) {
    vmdSqrt(n, in, out, VML_LA | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT);
    return;
}

#else // ifdef HAVE_MKL

void toast::vfast_sincos(int n, double const * ang, double * sinout,
                         double * cosout) {
    double const SC1 = 0.99999999999999806767;
    double const SC2 = -0.4999999999998996568;
    double const SC3 = 0.04166666666581174292;
    double const SC4 = -0.001388888886113613522;
    double const SC5 = 0.000024801582876042427;
    double const SC6 = -0.0000002755693576863181;
    double const SC7 = 0.0000000020858327958707;
    double const SC8 = -0.000000000011080716368;
    double sx;
    double cx;
    double sx2;
    double cx2;
    double ssign;
    double csign;
    double quot;
    double rem;
    double x;
    int quad;
    int i;

    # pragma \
    omp parallel for default(shared) private(i, sx, cx, sx2, cx2, ssign, csign, quot, rem, x, quad) schedule(static)
    for (i = 0; i < n; i++) {
        quot = ang[i] * toast::INV_TWOPI;
        rem = quot - floor(quot);
        x = rem * toast::TWOPI;
        while (x < 0.0) {
            x += toast::TWOPI;
        }
        quad = static_cast <int> (x * toast::TWOINVPI);
        switch (quad) {
            case 1:
                sx = x - toast::PI_2;
                ssign = 1.0;
                cx = toast::PI - x;
                csign = -1.0;
                break;

            case 2:
                sx = toast::THREEPI_2 - x;
                ssign = -1.0;
                cx = x - toast::PI;
                csign = -1.0;
                break;

            case 3:
                sx = x - toast::THREEPI_2;
                ssign = -1.0;
                cx = toast::TWOPI - x;
                csign = 1.0;
                break;

            default:
                sx = toast::PI_2 - x;
                ssign = 1.0;
                cx = x;
                csign = 1.0;
                break;
        }
        sx2 = sx * sx;
        cx2 = cx * cx;

        sinout[i] = ssign * (SC1 + sx2 *
                             (SC2 + sx2 *
                              (SC3 + sx2 *
                               (SC4 + sx2 *
                                (SC5 + sx2 *
                                 (SC6 + sx2 * (SC7 + sx2 * SC8)))))));
        cosout[i] = csign * (SC1 + cx2 *
                             (SC2 + cx2 *
                              (SC3 + cx2 *
                               (SC4 + cx2 *
                                (SC5 + cx2 *
                                 (SC6 + cx2 * (SC7 + cx2 * SC8)))))));
    }
    return;
}

void toast::vfast_atan2(int n, double const * y, double const * x,
                        double * ang) {
    double const ATCHEB1 = 48.70107004404898384;
    double const ATCHEB2 = 49.5326263772254345;
    double const ATCHEB3 = 9.40604244231624;
    double const ATCHEB4 = 48.70107004404996166;
    double const ATCHEB5 = 65.7663163908956299;
    double const ATCHEB6 = 21.587934067020262;
    int i;
    double r2;
    double r;
    int complement;
    int region;
    int sign;

    # pragma \
    omp parallel for default(shared) private(i, r, r2, complement, region, sign) schedule(static)
    for (i = 0; i < n; i++) {
        r = y[i] / x[i];

        // reduce range to PI/12

        complement = 0;
        region = 0;
        sign = 0;
        if (r < 0) {
            r = -r;
            sign = 1;
        }
        if (r > 1.0) {
            r = 1.0 / r;
            complement = 1;
        }
        if (r > toast::TANTWELFTHPI) {
            r = (r - toast::TANSIXTHPI) / (1 + toast::TANSIXTHPI * r);
            region = 1;
        }
        r2 = r * r;
        r = (r * (ATCHEB1 + r2 * (ATCHEB2 + r2 * ATCHEB3))) /
            (ATCHEB4 + r2 * (ATCHEB5 + r2 * (ATCHEB6 + r2)));
        if (region) {
            r += toast::SIXTHPI;
        }
        if (complement) {
            r = toast::PI_2 - r;
        }
        if (sign) {
            r = -r;
        }

        // adjust quadrant

        if (x[i] > 0.0) {
            ang[i] = r;
        } else if (x[i] < 0.0) {
            ang[i] = r + toast::PI;
            if (ang[i] > toast::PI) {
                ang[i] -= toast::TWOPI;
            }
        } else if (y[i] > 0.0) {
            ang[i] = toast::PI_2;
        } else if (y[i] < 0.0) {
            ang[i] = -toast::PI_2;
        } else {
            ang[i] = 0.0;
        }
    }
    return;
}

void toast::vfast_sqrt(int n, double const * in, double * out) {
    toast::vsqrt(n, in, out);
    return;
}

#endif // ifdef HAVE_MKL


// Quaternion operations needed for this test

void toast::qa_list_dot(size_t n, size_t m, size_t d, double const * a,
                        double const * b, double * dotprod) {
    if (toast::is_aligned(a) && toast::is_aligned(b) &&
        toast::is_aligned(dotprod)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            dotprod[i] = 0.0;
            size_t off = m * i;
            for (size_t j = 0; j < d; ++j) {
                dotprod[i] += a[off + j] * b[off + j];
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            dotprod[i] = 0.0;
            size_t off = m * i;
            for (size_t j = 0; j < d; ++j) {
                dotprod[i] += a[off + j] * b[off + j];
            }
        }
    }
    return;
}

void toast::qa_amplitude(size_t n, size_t m, size_t d, double const * v,
                         double * norm) {
    toast::AlignedVector <double> temp(n);

    toast::qa_list_dot(n, m, d, v, v, temp.data());

    toast::vsqrt(n, temp.data(), norm);

    return;
}

void toast::qa_normalize(size_t n, size_t m, size_t d,
                         double const * q_in, double * q_out) {
    toast::AlignedVector <double> norm(n);

    toast::qa_amplitude(n, m, d, q_in, norm.data());

    if (toast::is_aligned(q_in) && toast::is_aligned(q_out)) {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            size_t off = m * i;
            for (size_t j = 0; j < d; ++j) {
                q_out[off + j] = q_in[off + j] / norm[i];
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            size_t off = m * i;
            for (size_t j = 0; j < d; ++j) {
                q_out[off + j] = q_in[off + j] / norm[i];
            }
        }
    }

    return;
}

void toast::qa_rotate_many_one(size_t nq, double const * q,
                               double const * v_in, double * v_out) {
    toast::AlignedVector <double> q_unit(4 * nq);

    toast::qa_normalize(nq, 4, 4, q, q_unit.data());

    if (toast::is_aligned(v_in) && toast::is_aligned(v_out)) {
        #pragma omp simd
        for (size_t i = 0; i < nq; ++i) {
            size_t vfout = 3 * i;
            size_t qf = 4 * i;
            double xw =  q_unit[qf + 3] * q_unit[qf + 0];
            double yw =  q_unit[qf + 3] * q_unit[qf + 1];
            double zw =  q_unit[qf + 3] * q_unit[qf + 2];
            double x2 = -q_unit[qf + 0] * q_unit[qf + 0];
            double xy =  q_unit[qf + 0] * q_unit[qf + 1];
            double xz =  q_unit[qf + 0] * q_unit[qf + 2];
            double y2 = -q_unit[qf + 1] * q_unit[qf + 1];
            double yz =  q_unit[qf + 1] * q_unit[qf + 2];
            double z2 = -q_unit[qf + 2] * q_unit[qf + 2];

            v_out[vfout + 0] = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] +
                                    (yw + xz) * v_in[2]) + v_in[0];

            v_out[vfout + 1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] +
                                    (yz - xw) * v_in[2]) + v_in[1];

            v_out[vfout + 2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] +
                                    (x2 + y2) * v_in[2]) + v_in[2];
        }
    } else {
        for (size_t i = 0; i < nq; ++i) {
            size_t vfout = 3 * i;
            size_t qf = 4 * i;
            double xw =  q_unit[qf + 3] * q_unit[qf + 0];
            double yw =  q_unit[qf + 3] * q_unit[qf + 1];
            double zw =  q_unit[qf + 3] * q_unit[qf + 2];
            double x2 = -q_unit[qf + 0] * q_unit[qf + 0];
            double xy =  q_unit[qf + 0] * q_unit[qf + 1];
            double xz =  q_unit[qf + 0] * q_unit[qf + 2];
            double y2 = -q_unit[qf + 1] * q_unit[qf + 1];
            double yz =  q_unit[qf + 1] * q_unit[qf + 2];
            double z2 = -q_unit[qf + 2] * q_unit[qf + 2];

            v_out[vfout + 0] = 2 * ((y2 + z2) * v_in[0] + (xy - zw) * v_in[1] +
                                    (yw + xz) * v_in[2]) + v_in[0];

            v_out[vfout + 1] = 2 * ((zw + xy) * v_in[0] + (x2 + z2) * v_in[1] +
                                    (yz - xw) * v_in[2]) + v_in[1];

            v_out[vfout + 2] = 2 * ((xz - yw) * v_in[0] + (xw + yz) * v_in[1] +
                                    (x2 + y2) * v_in[2]) + v_in[2];
        }
    }

    return;
}

void toast::qa_mult_many_one(size_t np, double const * p,
                             double const * q, double * r) {
    if (toast::is_aligned(p) && toast::is_aligned(q) && toast::is_aligned(r)) {
        #pragma omp simd
        for (size_t i = 0; i < np; ++i) {
            size_t f = 4 * i;
            r[f + 0] =  p[f + 0] * q[3] + p[f + 1] * q[2] -
                       p[f + 2] * q[1] + p[f + 3] * q[0];
            r[f + 1] = -p[f + 0] * q[2] + p[f + 1] * q[3] +
                       p[f + 2] * q[0] + p[f + 3] * q[1];
            r[f + 2] =  p[f + 0] * q[1] - p[f + 1] * q[0] +
                       p[f + 2] * q[3] + p[f + 3] * q[2];
            r[f + 3] = -p[f + 0] * q[0] - p[f + 1] * q[1] -
                       p[f + 2] * q[2] + p[f + 3] * q[3];
        }
    } else {
        for (size_t i = 0; i < np; ++i) {
            size_t f = 4 * i;
            r[f + 0] =  p[f + 0] * q[3] + p[f + 1] * q[2] -
                       p[f + 2] * q[1] + p[f + 3] * q[0];
            r[f + 1] = -p[f + 0] * q[2] + p[f + 1] * q[3] +
                       p[f + 2] * q[0] + p[f + 3] * q[1];
            r[f + 2] =  p[f + 0] * q[1] - p[f + 1] * q[0] +
                       p[f + 2] * q[3] + p[f + 3] * q[2];
            r[f + 3] = -p[f + 0] * q[0] - p[f + 1] * q[1] -
                       p[f + 2] * q[2] + p[f + 3] * q[3];
        }
    }

    return;
}


// Healpix operations needed for this test.

const int64_t toast::HealpixPixels::jr_[] =
{2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};

const int64_t toast::HealpixPixels::jp_[] =
{1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};


void toast::HealpixPixels::init() {
    nside_ = 0;
    ncap_ = 0;
    npix_ = 0;
    dnside_ = 0.0;
    fournside_ = 0;
    twonside_ = 0;
    nsideplusone_ = 0;
    halfnside_ = 0.0;
    tqnside_ = 0.0;
    factor_ = 0;
    nsideminusone_ = 0;
}

toast::HealpixPixels::HealpixPixels() {
    init();
}

toast::HealpixPixels::HealpixPixels(int64_t nside) {
    init();
    reset(nside);
}

void toast::HealpixPixels::reset(int64_t nside) {
    if (nside <= 0) {
        std::string msg("cannot reset healpix pixels with NSIDE <= 0");
        throw std::runtime_error(msg.c_str());
    }

    // check for valid nside value

    uint64_t temp = static_cast <uint64_t> (nside);
    if (((~temp) & (temp - 1)) != (temp - 1)) {
        std::string msg("invalid NSIDE value- must be a multiple of 2");
        throw std::runtime_error(msg.c_str());
    }

    nside_ = nside;

    for (uint64_t m = 0; m < 0x100; ++m) {
        utab_[m] = (m & 0x1) | ((m & 0x2) << 1) | ((m & 0x4) << 2) |
                   ((m & 0x8) << 3) | ((m & 0x10) << 4) | ((m & 0x20) << 5) |
                   ((m & 0x40) << 6) | ((m & 0x80) << 7);

        ctab_[m] = (m & 0x1) | ((m & 0x2) << 7) | ((m & 0x4) >> 1) |
                   ((m & 0x8) << 6) | ((m & 0x10) >> 2) | ((m & 0x20) << 5) |
                   ((m & 0x40) >> 3) | ((m & 0x80) << 4);
    }

    ncap_ = 2 * (nside * nside - nside);

    npix_ = 12 * nside * nside;

    dnside_ = static_cast <double> (nside);

    twonside_ = 2 * nside;

    fournside_ = 4 * nside;

    nsideplusone_ = nside + 1;

    halfnside_ = 0.5 * (dnside_);

    tqnside_ = 0.75 * (dnside_);

    factor_ = 0;

    nsideminusone_ = nside - 1;

    while (nside != (1ll << factor_)) {
        ++factor_;
    }

    return;
}

void toast::HealpixPixels::vec2zphi(int64_t n, double const * vec,
                                    double * phi, int * region, double * z,
                                    double * rtz) const {
    if (n > std::numeric_limits <int>::max()) {
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        throw std::runtime_error(msg.c_str());
    }

    toast::AlignedVector <double> work1(n);
    toast::AlignedVector <double> work2(n);
    toast::AlignedVector <double> work3(n);

    if (toast::is_aligned(vec) && toast::is_aligned(phi) &&
        toast::is_aligned(region) && toast::is_aligned(z)
        && toast::is_aligned(rtz)) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            int64_t offset = 3 * i;

            // region encodes BOTH the sign of Z and whether its
            // absolute value is greater than 2/3.

            z[i] = vec[offset + 2];

            double za = ::fabs(z[i]);

            int itemp = (z[i] > 0.0) ? 1 : -1;

            region[i] = (za <= TWOTHIRDS) ? itemp : itemp + itemp;

            work1[i] = 3.0 * (1.0 - za);
            work3[i] = vec[offset + 1];
            work2[i] = vec[offset];
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            int64_t offset = 3 * i;

            // region encodes BOTH the sign of Z and whether its
            // absolute value is greater than 2/3.

            z[i] = vec[offset + 2];

            double za = ::fabs(z[i]);

            int itemp = (z[i] > 0.0) ? 1 : -1;

            region[i] = (za <= TWOTHIRDS) ? itemp : itemp + itemp;

            work1[i] = 3.0 * (1.0 - za);
            work3[i] = vec[offset + 1];
            work2[i] = vec[offset];
        }
    }

    toast::vfast_sqrt(n, work1.data(), rtz);
    toast::vfast_atan2(n, work3.data(), work2.data(), phi);

    return;
}

void toast::HealpixPixels::zphi2nest(int64_t n, double const * phi,
                                     int const * region, double const * z,
                                     double const * rtz, int64_t * pix) const {
    if (n > std::numeric_limits <int>::max()) {
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        throw std::runtime_error(msg.c_str());
    }
    if (toast::is_aligned(phi) && toast::is_aligned(pix) &&
        toast::is_aligned(region) && toast::is_aligned(z)
        && toast::is_aligned(rtz)) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            double tt =
                (phi[i] >= 0.0) ? phi[i] * TWOINVPI : phi[i] * TWOINVPI + 4.0;

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

            if (::abs(region[i]) == 1) {
                temp1 = halfnside_ + dnside_ * tt;
                temp2 = tqnside_ * z[i];

                jp = static_cast <int64_t> (temp1 - temp2);
                jm = static_cast <int64_t> (temp1 + temp2);

                ifp = jp >> factor_;
                ifm = jm >> factor_;

                if (ifp == ifm) {
                    face = (ifp == 4) ? static_cast <int64_t> (4) : ifp + 4;
                } else if (ifp < ifm) {
                    face = ifp;
                } else {
                    face = ifm + 8;
                }

                x = jm & nsideminusone_;
                y = nsideminusone_ - (jp & nsideminusone_);
            } else {
                ntt = static_cast <int64_t> (tt);

                tp = tt - static_cast <double> (ntt);

                temp1 = dnside_ * rtz[i];

                jp = static_cast <int64_t> (tp * temp1);
                jm = static_cast <int64_t> ((1.0 - tp) * temp1);

                if (jp >= nside_) {
                    jp = nsideminusone_;
                }
                if (jm >= nside_) {
                    jm = nsideminusone_;
                }

                if (z[i] >= 0) {
                    face = ntt;
                    x = nsideminusone_ - jm;
                    y = nsideminusone_ - jp;
                } else {
                    face = ntt + 8;
                    x = jp;
                    y = jm;
                }
            }

            uint64_t sipf = xy2pix_(static_cast <uint64_t> (x),
                                    static_cast <uint64_t> (y));

            pix[i] = static_cast <int64_t> (sipf) + (face << (2 * factor_));
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            double tt =
                (phi[i] >= 0.0) ? phi[i] * TWOINVPI : phi[i] * TWOINVPI + 4.0;

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

            if (::abs(region[i]) == 1) {
                temp1 = halfnside_ + dnside_ * tt;
                temp2 = tqnside_ * z[i];

                jp = static_cast <int64_t> (temp1 - temp2);
                jm = static_cast <int64_t> (temp1 + temp2);

                ifp = jp >> factor_;
                ifm = jm >> factor_;

                if (ifp == ifm) {
                    face = (ifp == 4) ? static_cast <int64_t> (4) : ifp + 4;
                } else if (ifp < ifm) {
                    face = ifp;
                } else {
                    face = ifm + 8;
                }

                x = jm & nsideminusone_;
                y = nsideminusone_ - (jp & nsideminusone_);
            } else {
                ntt = static_cast <int64_t> (tt);

                tp = tt - static_cast <double> (ntt);

                temp1 = dnside_ * rtz[i];

                jp = static_cast <int64_t> (tp * temp1);
                jm = static_cast <int64_t> ((1.0 - tp) * temp1);

                if (jp >= nside_) {
                    jp = nsideminusone_;
                }
                if (jm >= nside_) {
                    jm = nsideminusone_;
                }

                if (z[i] >= 0) {
                    face = ntt;
                    x = nsideminusone_ - jm;
                    y = nsideminusone_ - jp;
                } else {
                    face = ntt + 8;
                    x = jp;
                    y = jm;
                }
            }

            uint64_t sipf = xy2pix_(static_cast <uint64_t> (x),
                                    static_cast <uint64_t> (y));

            pix[i] = static_cast <int64_t> (sipf) + (face << (2 * factor_));
        }
    }

    return;
}

void toast::HealpixPixels::zphi2ring(int64_t n, double const * phi,
                                     int const * region, double const * z,
                                     double const * rtz, int64_t * pix) const {
    if (n > std::numeric_limits <int>::max()) {
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        throw std::runtime_error(msg.c_str());
    }
    if (toast::is_aligned(phi) && toast::is_aligned(pix) &&
        toast::is_aligned(region) && toast::is_aligned(z)
        && toast::is_aligned(rtz)) {
        #pragma omp simd
        for (int64_t i = 0; i < n; ++i) {
            double tt =
                (phi[i] >= 0.0) ? phi[i] * TWOINVPI : phi[i] * TWOINVPI + 4.0;

            double tp;
            int64_t longpart;
            double temp1;
            double temp2;
            int64_t jp;
            int64_t jm;
            int64_t ip;
            int64_t ir;
            int64_t kshift;

            if (::abs(region[i]) == 1) {
                temp1 = halfnside_ + dnside_ * tt;
                temp2 = tqnside_ * z[i];

                jp = static_cast <int64_t> (temp1 - temp2);
                jm = static_cast <int64_t> (temp1 + temp2);

                ir = nsideplusone_ + jp - jm;
                kshift = 1 - (ir & 1);

                ip = (jp + jm - nside_ + kshift + 1) >> 1;
                ip = ip % fournside_;

                pix[i] = ncap_ + ((ir - 1) * fournside_ + ip);
            } else {
                tp = tt - floor(tt);

                temp1 = dnside_ * rtz[i];

                jp = static_cast <int64_t> (tp * temp1);
                jm = static_cast <int64_t> ((1.0 - tp) * temp1);
                ir = jp + jm + 1;
                ip = static_cast <int64_t> (tt * (double)ir);
                longpart = static_cast <int64_t> (ip / (4 * ir));
                ip -= longpart;

                pix[i] = (region[i] > 0) ? (2 * ir * (ir - 1) + ip)
                         : (npix_ - 2 * ir * (ir + 1) + ip);
            }
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            double tt =
                (phi[i] >= 0.0) ? phi[i] * TWOINVPI : phi[i] * TWOINVPI + 4.0;

            double tp;
            int64_t longpart;
            double temp1;
            double temp2;
            int64_t jp;
            int64_t jm;
            int64_t ip;
            int64_t ir;
            int64_t kshift;

            if (::abs(region[i]) == 1) {
                temp1 = halfnside_ + dnside_ * tt;
                temp2 = tqnside_ * z[i];

                jp = static_cast <int64_t> (temp1 - temp2);
                jm = static_cast <int64_t> (temp1 + temp2);

                ir = nsideplusone_ + jp - jm;
                kshift = 1 - (ir & 1);

                ip = (jp + jm - nside_ + kshift + 1) >> 1;
                ip = ip % fournside_;

                pix[i] = ncap_ + ((ir - 1) * fournside_ + ip);
            } else {
                tp = tt - floor(tt);

                temp1 = dnside_ * rtz[i];

                jp = static_cast <int64_t> (tp * temp1);
                jm = static_cast <int64_t> ((1.0 - tp) * temp1);
                ir = jp + jm + 1;
                ip = static_cast <int64_t> (tt * (double)ir);
                longpart = static_cast <int64_t> (ip / (4 * ir));
                ip -= longpart;

                pix[i] = (region[i] > 0) ? (2 * ir * (ir - 1) + ip)
                         : (npix_ - 2 * ir * (ir + 1) + ip);
            }
        }
    }

    return;
}

void toast::HealpixPixels::vec2nest(int64_t n, double const * vec,
                                    int64_t * pix) const {
    if (n > std::numeric_limits <int>::max()) {
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        throw std::runtime_error(msg.c_str());
    }

    toast::AlignedVector <double> z(n);
    toast::AlignedVector <double> rtz(n);
    toast::AlignedVector <double> phi(n);
    toast::AlignedVector <int> region(n);

    vec2zphi(n, vec, phi.data(), region.data(), z.data(), rtz.data());

    zphi2nest(n, phi.data(), region.data(), z.data(), rtz.data(), pix);

    return;
}

void toast::HealpixPixels::vec2ring(int64_t n, double const * vec,
                                    int64_t * pix) const {
    if (n > std::numeric_limits <int>::max()) {
        std::string msg("healpix vector conversion must be in chunks of < 2^31");
        throw std::runtime_error(msg.c_str());
    }

    toast::AlignedVector <double> z(n);
    toast::AlignedVector <double> rtz(n);
    toast::AlignedVector <double> phi(n);
    toast::AlignedVector <int> region(n);

    vec2zphi(n, vec, phi.data(), region.data(), z.data(), rtz.data());

    zphi2ring(n, phi.data(), region.data(), z.data(), rtz.data(), pix);

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
        std::map <std::string, toast::AlignedVector <double> > & detweights,
        std::vector <double> & time_quat,
        std::vector <double> & time_pix,
        std::vector <double> & time_weight,
        std::vector <double> & time_tot) {

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

    #pragma \
    omp parallel default(none) shared(nsamp, ndet, nside, nest, boresight, hwpang, detnames, detquat, detcal, deteps, detpixels, detweights, time_quat, time_pix, time_weight, time_tot)
    {
        double xaxis[3] = {1.0, 0.0, 0.0};
        double zaxis[3] = {0.0, 0.0, 1.0};
        double nullquat[4] = {0.0, 0.0, 0.0, 1.0};
        toast::HealpixPixels hpix(nside);

        toast::Timer tmquat;
        toast::Timer tmpix;
        toast::Timer tmweight;
        toast::Timer tmtot;

        int trank = 1;
        #ifdef _OPENMP
        trank = omp_get_thread_num();
        #endif

        #pragma omp for schedule(dynamic)
        for (size_t d = 0; d < ndet; ++d) {
            tmtot.start();

            auto const & dname = detnames[d];
            auto const & cal = detcal.at(dname);
            auto const & eps = deteps.at(dname);
            auto const & quat = detquat.at(dname);

            toast::AlignedVector <int64_t> pixels(nsamp);
            toast::AlignedVector <double> weights(3 * nsamp);

            double eta = (1.0 - eps) / (1.0 + eps);

            tmquat.start();

            // Compute detector quaternions
            toast::AlignedVector <double> pntg(4 * nsamp);
            toast::qa_mult_many_one(nsamp, boresight.data(), quat.data(),
                                    pntg.data());

            // Direction vector
            toast::AlignedVector <double> dir(3 * nsamp);
            toast::qa_rotate_many_one(nsamp, pntg.data(), zaxis,
                                      dir.data());

            tmquat.stop();

            tmpix.start();

            // Sky pixel
            if (nest) {
                hpix.vec2nest(nsamp, dir.data(), pixels.data());
            } else {
                hpix.vec2ring(nsamp, dir.data(), pixels.data());
            }

            tmpix.stop();

            tmweight.start();

            // Orientation vector
            toast::AlignedVector <double> orient(3 * nsamp);
            toast::qa_rotate_many_one(nsamp, pntg.data(), xaxis,
                                      orient.data());

            // Workspace buffers that are re-used
            toast::AlignedVector <double> buf1(nsamp);
            toast::AlignedVector <double> buf2(nsamp);

            // Compute the angle of the polarization response with respect
            // to the local meridian.
            double * bx = buf1.data();
            double * by = buf2.data();

            #pragma omp simd
            for (size_t i = 0; i < nsamp; ++i) {
                size_t off = 3 * i;
                by[i] = orient[off + 0] * dir[off + 1] - orient[off + 1] *
                        dir[off + 0];
                bx[i] = orient[off + 0] * (-dir[off + 2] * dir[off + 0]) +
                        orient[off + 1] * (-dir[off + 2] * dir[off + 1]) +
                        orient[off + 2] * (dir[off + 0] * dir[off + 0] +
                                           dir[off + 1] * dir[off + 1]);
            }

            toast::AlignedVector <double> detang(nsamp);
            toast::vfast_atan2(nsamp, by, bx, detang.data());

            #pragma omp simd
            for (size_t i = 0; i < nsamp; ++i) {
                detang[i] += 2.0 * hwpang[i];
                detang[i] *= 2.0;
            }

            // Compute the Stokes weights

            double * sinout = buf1.data();
            double * cosout = buf2.data();

            toast::vfast_sincos(nsamp, detang.data(), sinout, cosout);

            for (size_t i = 0; i < nsamp; ++i) {
                size_t off = 3 * i;
                weights[off + 0] = cal;
                weights[off + 1] = cosout[i] * eta * cal;
                weights[off + 2] = sinout[i] * eta * cal;
            }

            tmweight.stop();

            #pragma omp critical
            {
                std::copy(pixels.begin(), pixels.end(), detpixels.at(dname).begin());
                std::copy(weights.begin(), weights.end(), detweights.at(dname).begin());
            }

            tmtot.stop();

            time_quat[trank] += tmquat.seconds();
            tmquat.clear();
            time_pix[trank] += tmpix.seconds();
            tmpix.clear();
            time_weight[trank] += tmweight.seconds();
            tmweight.clear();
            time_tot[trank] += tmtot.seconds();
            tmtot.clear();
        }
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

    int nthreads = omp_get_max_threads();

    std::vector <double> time_quat(nthreads);
    time_quat.assign(nthreads, 0.0);

    std::vector <double> time_pix(nthreads);
    time_pix.assign(nthreads, 0.0);

    std::vector <double> time_weight(nthreads);
    time_weight.assign(nthreads, 0.0);

    std::vector <double> time_tot(nthreads);
    time_tot.assign(nthreads, 0.0);

    for (size_t ob = 0; ob < nobs; ++ob) {
        toast::detector_pointing_healpix(nside, nest,
                                         boresight, hwpang,
                                         detnames, detquat,
                                         detcal, deteps,
                                         detpixels, detweights,
                                         time_quat, time_pix,
                                         time_weight, time_tot);
    }

    for (int i = 0; i < nthreads; ++i) {
        std::cout << std::setprecision(2) << std::fixed
            << "Thread " << i << ":" << std::endl
            << "  compute detector quaternions: " << time_quat[i] << " s"
            << std::endl
            << "  compute pixel numbers: " << time_pix[i] << " s"
            << std::endl
            << "  compute Stokes weights: " << time_weight[i] << " s"
            << std::endl
            << "  total: " << time_tot[i] << " s" << std::endl;
    }

    return;
}
