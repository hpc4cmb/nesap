
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
        int numSMs, cudaStream_t * streams,
        std::map <std::string, toast::AlignedVector <int64_t> > & detpixels,
        std::map <std::string, toast::AlignedVector <double> > & detweights);

}

#endif
