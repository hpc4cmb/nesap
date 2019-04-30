
#ifndef POINTING_CUDA_CUH
#define POINTING_CUDA_CUH

#include <cstddef>
#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>


static void CudaError(cudaError_t err, char const * file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(err) (CudaError(err, __FILE__, __LINE__))


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


namespace toast {

// High-level pointing function.

void detector_pointing_healpix(
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
    std::map <std::string, toast::AlignedVector <double> > & detweights);

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
