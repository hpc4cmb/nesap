
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

            n_input[k + 1] = fftlen;
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

    // Forward FFTs

    fftw_execute(fplan);

    // Convolve with kernel

    for (int k = 0; k < nffts; ++k) {
        int bufoff = k * fftlen;
        for (int i = 0; i < fftlen; ++i ) {
            rdata[bufoff + i] *= filter[i];
        }
    }

    // Reverse transform

    fftw_execute(rplan);

    // Copy back to TOD buffer

    for (int k = 0; k < nffts; ++k) {
        int bufoff = k * fftlen;
        std::copy(
            &(fdata[bufoff + off_outfft[k]]),
            &(fdata[bufoff + off_outfft[k] + n_output[k]]),
            &(tod[off_outdata[k]]));
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
        int fftlen, int nffts, int ncore, int nmiddle, int overlap,
        fftw_plan * fplans, fftw_plan * rplans,
        std::map <int, toast::AlignedVector <double> > & tfdata,
        std::map <int, toast::AlignedVector <double> > & trdata,
        int64_t nsubmap, int64_t nnz, std::map <int64_t, int64_t> const & smlocal,
        toast::AlignedVector <double> & result,
        std::vector <double> & time_pquat,
        std::vector <double> & time_ppix,
        std::vector <double> & time_pweight,
        std::vector <double> & time_ptot,
        std::vector <double> & time_amult,
        std::vector <double> & time_atmult,
        std::vector <double> & time_nmult,
        std::vector <double> & time_tot
    ) {

    size_t nsamp = (size_t)(boresight.size() / 4);

    size_t ndet = detnames.size();

    #pragma \
    omp parallel default(none) shared(nsamp, ndet, nside, nest, boresight, hwpang, filter, detnames, detquat, detcal, deteps, fftlen, nffts, ncore, nmiddle, overlap, fplans, rplans, tfdata, trdata, nsubmap, nnz, smlocal, result, time_pquat, time_ppix, time_pweight, time_ptot, time_amult, time_atmult, time_nmult, time_tot, std::cerr)
    {
        toast::HealpixPixels hpix(nside);

        toast::Timer tmpquat;
        toast::Timer tmppix;
        toast::Timer tmpweight;
        toast::Timer tmptot;
        toast::Timer tmamult;
        toast::Timer tmatmult;
        toast::Timer tmnmult;
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
            toast::AlignedVector <double> local_result(result);

            // Compute detector pointing

            tmptot.start();

            toast::single_detector(nest, hpix, cal, eps, quat, nsamp, hwpang,
                                   boresight, pixels, weights, tmpquat, tmppix,
                                   tmpweight);

            tmptot.stop();

            // Convert global pixels to local pixels

            for (size_t i = 0; i < nsamp; ++i) {
                int64_t gsm = pixels[i] / nsubmap;
                int64_t smpix = pixels[i] % nsubmap;
                int64_t lsm = smlocal.at(gsm);
                pixels[i] = (lsm * nsubmap) + smpix;
            }

            // Sample from starting map to create timestream

            tmamult.start();

            toast::AlignedVector <double> tod(nsamp);
            std::fill(tod.begin(), tod.end(), 0.0);

            for (size_t i = 0; i < nsamp; ++i) {
                int64_t poff = nnz * pixels[i];
                int64_t toff = nnz * i;
                for (int64_t j = 0; j < nnz; ++j) {
                    tod[i] = weights[toff + j] * local_result[poff + j];
                }
            }

            tmamult.stop();

            // Apply Toeplitz noise covariance to TOD.

            tmnmult.start();

            toeplitz_multiply(
                    fftlen, nffts, ncore, nmiddle, overlap,
                    fplans[trank], rplans[trank], tfdata[trank], trdata[trank],
                    filter, tod);

            tmnmult.stop();

            // Accumulate to result

            tmatmult.start();

            std::fill(local_result.begin(), local_result.end(), 0.0);

            for (size_t i = 0; i < nsamp; ++i) {
                int64_t poff = nnz * pixels[i];
                int64_t toff = nnz * i;
                for (int64_t j = 0; j < nnz; ++j) {
                    local_result[poff + j] += weights[toff + j] * tod[i];
                }
            }

            tmatmult.stop();

            tmtot.stop();

            time_pquat[trank] += tmpquat.seconds();
            tmpquat.clear();

            time_ppix[trank] += tmppix.seconds();
            tmppix.clear();

            time_pweight[trank] += tmpweight.seconds();
            tmpweight.clear();

            time_ptot[trank] += tmptot.seconds();
            tmptot.clear();

            time_amult[trank] += tmamult.seconds();
            tmamult.clear();

            time_atmult[trank] += tmatmult.seconds();
            tmatmult.clear();

            time_nmult[trank] += tmnmult.seconds();
            tmnmult.clear();

            time_tot[trank] += tmtot.seconds();
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

    auto & gt = toast::GlobalTimers::get();

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

    std::vector <double> time_amult(nthreads);
    time_amult.assign(nthreads, 0.0);

    std::vector <double> time_atmult(nthreads);
    time_atmult.assign(nthreads, 0.0);

    std::vector <double> time_nmult(nthreads);
    time_nmult.assign(nthreads, 0.0);

    std::vector <double> time_tot(nthreads);
    time_tot.assign(nthreads, 0.0);

    // First we must pass through the pointing once in order to build up the locally
    // hit pixels.

    // Use a typical NSIDE=16 value
    int64_t nsubmap = 12 * 16 * 16;

    // We have Stokes I/Q/U values.
    int64_t nnz = 3;

    std::set <int64_t> submaps;
    submaps.clear();

    gt.start("Calculate local pixel distribution");

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
        detector_pointing_healpix(nside, nest, boresight, hwpang,
                                  detnames, detquat, detcal, deteps,
                                  detpixels, detweights,
                                  time_pquat, time_ppix,
                                  time_pweight, time_ptot);
        for (auto const & dname : detnames) {
            toast::update_submaps(nsubmap, detpixels[dname], submaps);
        }
        std::cerr << "Compute locally hit pixels:  stop observation " << ob << std::endl;
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

    gt.stop("Calculate local pixel distribution");

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

    gt.start("Calculate one iteration of LHS");

    for (size_t ob = 0; ob < nobs; ++ob) {
        std::cerr << "Compute solver LHS:  start observation " << ob << std::endl;
        solver_lhs_obs(
            nside, nest, boresight, hwpang, filtkern, detnames,
            detquat, detcal, deteps, fftlen, nffts, ncore, nmiddle, overlap,
            fplans, rplans, tfdata, trdata, nsubmap, nnz, smlocal, result,
            time_pquat, time_ppix, time_pweight, time_ptot, time_amult, time_atmult,
            time_nmult, time_tot
        );
        std::cerr << "Compute solver LHS:  stop observation " << ob << std::endl;
    }

    gt.stop("Calculate one iteration of LHS");

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
            << "  pointing total: " << time_ptot[i] << " s" << std::endl
            << "  A multiply: " << time_amult[i] << " s" << std::endl
            << "  N^1 multiply: " << time_nmult[i] << " s" << std::endl
            << "  A^T multiply: " << time_atmult[i] << " s" << std::endl
            << "  Total: " << time_tot[i] << " s" << std::endl;
    }

    return;
}
