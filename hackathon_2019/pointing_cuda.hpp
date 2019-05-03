
#ifndef POINTING_CUDA_HPP
#define POINTING_CUDA_HPP

#include <cstddef>
#include <cmath>
#include <cstdio>


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

void qa_normalize_inplace(size_t n, double * q);

void single_detector_nest(
        hpix * hp,
        double cal,
        double eps,
        double const * detquat,
        int nsamp,
        double const * hwpang,
        double const * boresight,
        int64_t * detpixels,
        float * detweights
    );

 void single_detector_ring(
        hpix * hp,
        double cal,
        double eps,
        double const * detquat,
        int nsamp,
        double const * hwpang,
        double const * boresight,
        int64_t * detpixels,
        float * detweights
    );

namespace toast {

// High-level pointing function.

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
