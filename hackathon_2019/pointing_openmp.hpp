
#ifndef POINTING_OPENMP_HPP
#define POINTING_OPENMP_HPP

#include <cstddef>
#include <cmath>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <memory>
#include <map>
#include <vector>


namespace toast {

// Vector math needed for this test

void vsqrt(int n, double const * in, double * out);
void vfast_sincos(int n, double const * ang, double * sinout, double * cosout);
void vfast_atan2(int n, double const * y, double const * x, double * ang);
void vfast_sqrt(int n, double const * in, double * out);

// Quaternion operations (only the ones we need for this test)

void qa_list_dot(size_t n, size_t m, size_t d, double const * a,
                 double const * b, double * dotprod);
void qa_amplitude(size_t n, size_t m, size_t d, double const * v,
                  double * norm);
void qa_normalize(size_t n, size_t m, size_t d,
                  double const * q_in, double * q_out);
void qa_rotate_many_one(size_t nq, double const * q,
                        double const * v_in, double * v_out);
void qa_mult_many_one(size_t np, double const * p,
                      double const * q, double * r);

// Healpix operations (only the ones we need for this test)

class HealpixPixels {
    public:

        HealpixPixels();
        HealpixPixels(int64_t nside);
        ~HealpixPixels() {}

        void reset(int64_t nside);

        void vec2zphi(int64_t n, double const * vec, double * phi,
                      int * region, double * z, double * rtz) const;

        void zphi2nest(int64_t n, double const * phi, int const * region,
                       double const * z, double const * rtz,
                       int64_t * pix) const;

        void zphi2ring(int64_t n, double const * phi, int const * region,
                       double const * z, double const * rtz,
                       int64_t * pix) const;

        void vec2nest(int64_t n, double const * vec, int64_t * pix) const;

        void vec2ring(int64_t n, double const * vec, int64_t * pix) const;

    private:

        void init();

        uint64_t xy2pix_(uint64_t x, uint64_t y) const {
            return utab_[x & 0xff] | (utab_[(x >> 8) & 0xff] << 16) |
                   (utab_[(x >> 16) & 0xff] << 32) |
                   (utab_[(x >> 24) & 0xff] << 48) |
                   (utab_[y & 0xff] << 1) | (utab_[(y >> 8) & 0xff] << 17) |
                   (utab_[(y >> 16) & 0xff] << 33) |
                   (utab_[(y >> 24) & 0xff] << 49);
        }

        uint64_t x2pix_(uint64_t x) const {
            return utab_[x & 0xff] | (utab_[x >> 8] << 16) |
                   (utab_[(x >> 16) & 0xff] << 32) |
                   (utab_[(x >> 24) & 0xff] << 48);
        }

        uint64_t y2pix_(uint64_t y) const {
            return (utab_[y & 0xff] << 1) | (utab_[y >> 8] << 17) |
                   (utab_[(y >> 16) & 0xff] << 33) |
                   (utab_[(y >> 24) & 0xff] << 49);
        }

        void pix2xy_(uint64_t pix, uint64_t & x, uint64_t & y) const {
            uint64_t raw;
            raw = (pix & 0x5555ull) | ((pix & 0x55550000ull) >> 15) |
                  ((pix & 0x555500000000ull) >> 16) |
                  ((pix & 0x5555000000000000ull) >> 31);
            x = ctab_[raw & 0xff] | (ctab_[(raw >> 8) & 0xff] << 4) |
                (ctab_[(raw >> 16) & 0xff] << 16) |
                (ctab_[(raw >> 24) & 0xff] << 20);
            raw = ((pix & 0xaaaaull) >> 1) | ((pix & 0xaaaa0000ull) >> 16) |
                  ((pix & 0xaaaa00000000ull) >> 17) |
                  ((pix & 0xaaaa000000000000ull) >> 32);
            y = ctab_[raw & 0xff] | (ctab_[(raw >> 8) & 0xff] << 4) |
                (ctab_[(raw >> 16) & 0xff] << 16) |
                (ctab_[(raw >> 24) & 0xff] << 20);
            return;
        }

        static const int64_t jr_[];
        static const int64_t jp_[];
        uint64_t utab_[0x100];
        uint64_t ctab_[0x100];
        int64_t nside_;
        int64_t npix_;
        int64_t ncap_;
        double dnside_;
        int64_t twonside_;
        int64_t fournside_;
        int64_t nsideplusone_;
        int64_t nsideminusone_;
        double halfnside_;
        double tqnside_;
        int64_t factor_;
};


// High-level pointing function.

void detector_pointing_healpix(
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
        std::vector <double> & time_tot);

void pointing(
    int64_t nside, bool nest,
    toast::AlignedVector <double> const & boresight,
    toast::AlignedVector <double> const & hwpang,
    toast::AlignedVector <std::string> const & detnames,
    std::map <std::string, toast::AlignedVector <double> > const & detquat,
    std::map <std::string, double> const & detcal,
    std::map <std::string, double> const & deteps,
    std::map <std::string, toast::AlignedVector <int64_t> > & detpixels,
    std::map <std::string, toast::AlignedVector <double> > & detweights, size_t nobs);

}

#endif
