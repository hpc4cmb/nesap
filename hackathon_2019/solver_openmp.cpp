
#include <utils.hpp>
#include <pointing_openmp.hpp>
#include <solver_openmp.hpp>

#include <cmath>
#include <sstream>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <fftw3.h>


void toeplitz_multiply(
        int fftlen, int nffts, int ncore, int nmiddle, int overlap,
        fftw_plan & fplan, fftw_plan & rplan,
        toast::AlignedVector <double> & fdata, toast::AlignedVector <double> & rdata,
        toast::AlignedVector <double> const & filter,
        toast::AlignedVector <double> & tod) {
    // Note:  TOD buffer is replaced by output.
    // We use "int" everywhere here since all the FFT math libraries use those.
    // We would never take an FFT of 2^31 samples...

    int nsamp = (int)tod.size();

    // Clear the input buffer
    std::fill(fdata.begin(), fdata.end(), 0.0);

    std::vector <int> n_input(nffts);
    std::vector <int> off_indata(nffts);
    std::vector <int> off_infft(nffts);
    std::vector <int> n_output(nffts);
    std::vector <int> off_outdata(nffts);
    std::vector <int> off_outfft(nffts);

    int trank = 1;
    #ifdef _OPENMP
    trank = omp_get_thread_num();
    #endif

    std::ostringstream msg;
    msg.str("");
    msg << "    thread " << trank << " start input copy";
    std::cerr << msg.str() << std::endl;

    if (nffts == 1) {
        // one shot
        n_input[0] = nsamp;
        off_indata[0] = 0;
        off_infft[0] = (fftlen - nsamp) >> 1;

        n_output[0] = nsamp;
        off_outdata[0] = 0;
        off_outfft[0] = off_infft[0];

        int bufoff = 0;

        std::copy(&(tod[off_indata[0]]), &(tod[off_indata[0] + n_input[0]]),
                  &(fdata[bufoff + off_infft[0]]));
    } else {
        // first fft
        n_input[0] = fftlen - overlap;
        if (n_input[0] > nsamp) {
            n_input[0] = nsamp;
        }
        off_indata[0] = 0;
        off_infft[0] = overlap;

        n_output[0] = ncore;
        off_outdata[0] = 0;
        off_outfft[0] = overlap;

        int bufoff = 0;

        std::copy(&(tod[off_indata[0]]), &(tod[off_indata[0] + n_input[0]]),
                  &(fdata[bufoff + off_infft[0]]));

        // middle ffts

        for (int k = 0; k < nmiddle; ++k) {
            n_output[k + 1] = ncore;
            off_outdata[k + 1] = (int)((nsamp - (nmiddle * ncore)) / 2) + k * ncore;
            off_outfft[k + 1] = overlap;

            n_input[k + 1] = nffts;
            if (overlap > off_outdata[k + 1]) {
                off_indata[k + 1] = 0;
            } else {
                off_indata[k + 1] = off_outdata[k + 1] - overlap;
            }
            off_infft[k + 1] = 0;

            bufoff = (k + 1) * fftlen;
            std::copy(
                &(tod[off_indata[k + 1]]),
                &(tod[off_indata[k + 1] + n_input[k + 1]]),
                &(fdata[bufoff + off_infft[k + 1]]));
        }

        // last fft
        n_input[nffts - 1] = fftlen - overlap;
        if (n_input[nffts - 1] > nsamp) {
            n_input[nffts - 1] = nsamp;
        }
        off_indata[nffts - 1] = nsamp - n_input[nffts - 1];
        off_infft[nffts - 1] = 0;

        n_output[nffts - 1] = ncore;
        off_outdata[nffts - 1] = nsamp - n_output[nffts - 1];
        off_outfft[nffts - 1] = overlap;

        bufoff = (nffts - 1) * fftlen;

        std::copy(
            &(tod[off_indata[nffts - 1]]),
            &(tod[off_indata[nffts - 1] + n_input[nffts - 1]]),
            &(fdata[bufoff + off_infft[nffts - 1]]));
    }

    msg.str("");
    msg << "    thread " << trank << " stop input copy";
    std::cerr << msg.str() << std::endl;

    // Forward FFTs

    msg.str("");
    msg << "    thread " << trank << " start forward fft";
    std::cerr << msg.str() << std::endl;

    fftw_execute(fplan);

    msg.str("");
    msg << "    thread " << trank << " stop forward fft";
    std::cerr << msg.str() << std::endl;

    // Convolve with kernel

    msg.str("");
    msg << "    thread " << trank << " start convolve";
    std::cerr << msg.str() << std::endl;

    for (int k = 0; k < nffts; ++k) {
        int bufoff = k * fftlen;
        for (int i = 0; i < fftlen; ++i ) {
            rdata[bufoff + i] *= filter[i];
        }
    }

    msg.str("");
    msg << "    thread " << trank << " stop convolve";
    std::cerr << msg.str() << std::endl;

    // Reverse transform

    msg.str("");
    msg << "    thread " << trank << " start reverse transform";
    std::cerr << msg.str() << std::endl;

    fftw_execute(rplan);

    msg.str("");
    msg << "    thread " << trank << " start reverse transform";
    std::cerr << msg.str() << std::endl;

    // Copy back to TOD buffer

    msg.str("");
    msg << "    thread " << trank << " start output copy";
    std::cerr << msg.str() << std::endl;

    for (int k = 0; k < nffts; ++k) {
        int bufoff = k * fftlen;
        std::copy(
            &(fdata[bufoff + off_outfft[k]]),
            &(fdata[bufoff + off_outfft[k] + n_output[k]]),
            &(tod[off_outdata[k]]));
    }

    msg.str("");
    msg << "    thread " << trank << " stop output copy";
    std::cerr << msg.str() << std::endl;

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
        int fftlen, int nffts, int ncore, int nmiddle, int overlap,
        fftw_plan * fplans, fftw_plan * rplans,
        std::map <int, toast::AlignedVector <double> > & tfdata,
        std::map <int, toast::AlignedVector <double> > & trdata,
        int64_t nsubmap, int64_t nnz, std::map <int64_t, int64_t> const & smlocal,
        toast::AlignedVector <double> & result,
        std::vector <double> & time_pquat,
        std::vector <double> & time_ppix,
        std::vector <double> & time_pweight,
        std::vector <double> & time_ptot) {

    size_t nsamp = (size_t)(boresight.size() / 4);

    size_t ndet = detnames.size();

    std::map <std::string, toast::AlignedVector <int64_t> > detpixels;
    std::map <std::string, toast::AlignedVector <double> > detweights;

    for (auto const & dname : detnames) {
        detpixels[dname].clear();
        detweights[dname].clear();
        detpixels[dname].resize(hwpang.size());
        detweights[dname].resize(3 * hwpang.size());
    }

    #pragma \
    omp parallel default(none) shared(nsamp, ndet, nside, nest, boresight, hwpang, filter, detnames, detquat, detcal, deteps, fftlen, nffts, ncore, nmiddle, overlap, fplans, rplans, tfdata, trdata, nsubmap, nnz, smlocal, detpixels, detweights, result, time_pquat, time_ppix, time_pweight, time_ptot, std::cerr)
    {
        toast::HealpixPixels hpix(nside);

        toast::Timer tmquat;
        toast::Timer tmpix;
        toast::Timer tmweight;
        toast::Timer tmtot;

        std::ostringstream msg;

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
            toast::AlignedVector <float> weights(3 * nsamp);

            // Compute detector pointing

            msg.str("");
            msg << "  det " << dname << " start pointing";
            std::cerr << msg.str() << std::endl;

            toast::single_detector(nest, hpix, cal, eps, quat, nsamp, hwpang,
                                   boresight, pixels, weights, tmquat, tmpix, tmweight);

            msg.str("");
            msg << "  det " << dname << " stop pointing";
            std::cerr << msg.str() << std::endl;

            // Sample from starting map to create timestream

            msg.str("");
            msg << "  det " << dname << " start A";
            std::cerr << msg.str() << std::endl;

            toast::AlignedVector <double> tod(nsamp);
            std::fill(tod.begin(), tod.end(), 0.0);

            for (size_t i = 0; i < nsamp; ++i) {
                if (i % 1000 == 0) {
                    msg.str("");
                    msg << "    det " << dname << " passing " << i;
                    std::cerr << msg.str() << std::endl;
                }
                int64_t lsm = smlocal.at(pixels[i] / nsubmap);
                int64_t lpix = pixels[i] % nsubmap;
                int64_t poff = nnz * ((lsm * nsubmap) + lpix);
                int64_t toff = nnz * i;
                for (int64_t j = 0; j < nnz; ++nnz) {
                    tod[i] = weights[toff + j] * result[poff + j];
                }
            }

            msg.str("");
            msg << "  det " << dname << " stop A";
            std::cerr << msg.str() << std::endl;

            // Apply Toeplitz noise covariance to TOD.

            msg.str("");
            msg << "  det " << dname << " start N^1";
            std::cerr << msg.str() << std::endl;

            toeplitz_multiply(
                    fftlen, nffts, ncore, nmiddle, overlap,
                    fplans[trank], rplans[trank], tfdata[trank], trdata[trank],
                    filter, tod);

            msg.str("");
            msg << "  det " << dname << " stop N^1";
            std::cerr << msg.str() << std::endl;

            // Accumulate to result

            std::fill(result.begin(), result.end(), 0.0);

            msg.str("");
            msg << "  det " << dname << " start A^T";
            std::cerr << msg.str() << std::endl;

            for (size_t i = 0; i < nsamp; ++i) {
                int64_t lsm = smlocal.at(pixels[i] / nsubmap);
                int64_t lpix = pixels[i] % nsubmap;
                int64_t poff = nnz * ((lsm * nsubmap) + lpix);
                int64_t toff = nnz * i;
                for (int64_t j = 0; j < nnz; ++nnz) {
                    result[poff + j] += weights[toff + j] * tod[i];
                }
            }

            msg.str("");
            msg << "  det " << dname << " stop A^T";
            std::cerr << msg.str() << std::endl;

            tmtot.stop();

            time_pquat[trank] += tmquat.seconds();
            tmquat.clear();

            time_ppix[trank] += tmpix.seconds();
            tmpix.clear();

            time_pweight[trank] += tmweight.seconds();
            tmweight.clear();

            time_ptot[trank] += tmtot.seconds();
            tmtot.clear();
        }
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

    int nthreads = omp_get_max_threads();

    std::vector <double> time_pquat(nthreads);
    time_pquat.assign(nthreads, 0.0);

    std::vector <double> time_ppix(nthreads);
    time_ppix.assign(nthreads, 0.0);

    std::vector <double> time_pweight(nthreads);
    time_pweight.assign(nthreads, 0.0);

    std::vector <double> time_ptot(nthreads);
    time_ptot.assign(nthreads, 0.0);

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
        std::cerr << "start submap observation " << ob << std::endl;
        detector_pointing_healpix(nside, nest, boresight, hwpang,
                                  detnames, detquat, detcal, deteps,
                                  detpixels, detweights,
                                  time_pquat, time_ppix,
                                  time_pweight, time_ptot);
        for (auto const & dname : detnames) {
            toast::update_submaps(nsubmap, detpixels[dname], submaps);
        }
        std::cerr << "stop submap observation " << ob << std::endl;
    }

    detpixels.clear();
    detweights.clear();

    // Now allocate the result map
    int64_t nsmlocal = submaps.size();
    std::map <int64_t, int64_t> smlocal;
    int64_t sm = 0;
    for (auto const & smap : submaps) {
        smlocal[smap] = sm;
        sm++;
    }
    result.resize(nnz * nsubmap * nsmlocal);
    std::fill(result.begin(), result.end(), 1.0);

    fftw_plan_with_nthreads(nthreads);

    int npsd = filter.size();
    int half = npsd - 1;
    int fftlen = 4 * half;

    // NOTE:  we are "cheating" here since we know that observations are all the same
    // length.  Normally we would cache all the plan lengths and batch sizes that are
    // used throughout the code...

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

    fftw_r2r_kind kind;

    unsigned flags = 0;
    flags |= FFTW_DESTROY_INPUT;
    flags |= FFTW_ESTIMATE;

    std::map <int, toast::AlignedVector <double> > tfdata;
    std::map <int, toast::AlignedVector <double> > trdata;

    fftw_plan fplans[nthreads];
    fftw_plan rplans[nthreads];

    for (int i = 0; i < nthreads; ++i) {
        tfdata[i].resize(nffts * fftlen);
        trdata[i].resize(nffts * fftlen);
        kind = FFTW_R2HC;
        fplans[i] = fftw_plan_many_r2r(
            1, &fftlen, nffts, tfdata.at(i).data(), &fftlen,
            1, fftlen, trdata.at(i).data(), &fftlen, 1, fftlen, &kind, flags);
        kind = FFTW_HC2R;
        rplans[i] = fftw_plan_many_r2r(
            1, &fftlen, nffts, trdata.at(i).data(), &fftlen,
            1, fftlen, tfdata.at(i).data(), &fftlen, 1, fftlen, &kind, flags);
    }

    for (size_t ob = 0; ob < nobs; ++ob) {
        std::cerr << "start solver_lhs_obs " << ob << std::endl;
        solver_lhs_obs(
            nside, nest, boresight, hwpang, filtkern, detnames,
            detquat, detcal, deteps, fftlen, nffts, ncore, nmiddle, overlap,
            fplans, rplans, tfdata, trdata, nsubmap, nnz, smlocal, result,
            time_pquat, time_ppix, time_pweight, time_ptot
        );
        std::cerr << "stop solver_lhs_obs " << ob << std::endl;
    }

    for (int i = 0; i < nthreads; ++i) {
        fftw_destroy_plan(fplans[i]);
        fftw_destroy_plan(rplans[i]);
    }

    for (int i = 0; i < nthreads; ++i) {
        std::cout << std::setprecision(2) << std::fixed
            << "Thread " << i << ":" << std::endl
            << "  compute detector quaternions: " << time_pquat[i] << " s"
            << std::endl
            << "  compute pixel numbers: " << time_ppix[i] << " s"
            << std::endl
            << "  compute Stokes weights: " << time_pweight[i] << " s"
            << std::endl
            << "  total: " << time_ptot[i] << " s" << std::endl;
    }

    return;
}
