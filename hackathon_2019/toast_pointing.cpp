
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>

#include <utils.hpp>
#include <pointing_openmp.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif


int main(int argc, char * argv[]) {
    auto & gt = toast::GlobalTimers::get();

    // Number of observations
    size_t nobs = 90;

    // Read input data.

    if (argc != 4) {
        std::cerr << "Usage:  " << argv[0]
            << " <focalplane file> <boresight file> <check file>" << std::endl;
        return 1;
    }

    gt.start("Read inputs");

    std::string fpfile(argv[1]);
    std::string borefile(argv[2]);
    std::string checkfile(argv[3]);

    toast::AlignedVector <std::string> detnames;
    std::map <std::string, toast::AlignedVector <double> > detquat;
    std::map <std::string, double> detcal;
    std::map <std::string, double> deteps;

    {
        std::ifstream handle(fpfile);
        std::string line;
        while (std::getline(handle, line)) {
            std::istringstream iss(line);
            std::string dname;
            double q0;
            double q1;
            double q2;
            double q3;
            if (!(iss >> dname >> q0 >> q1 >> q2 >> q3)) {
                break;
            }
            detnames.push_back(dname);
            detquat[dname] = toast::AlignedVector <double> (4);
            detquat[dname][0] = q0;
            detquat[dname][1] = q1;
            detquat[dname][2] = q2;
            detquat[dname][3] = q3;
            detcal[dname] = 1.0;
            deteps[dname] = 0.0;
        }
    }

    toast::AlignedVector <double> boresight;
    toast::AlignedVector <double> hwpang;

    double hwpdelta = 4.0 * toast::PI / 100.0;

    {
        std::ifstream handle(borefile);
        std::string line;
        double hwpoff = 0.0;
        while (std::getline(handle, line)) {
            std::istringstream iss(line);
            double q0;
            double q1;
            double q2;
            double q3;
            if (!(iss >> q0 >> q1 >> q2 >> q3)) {
                break;
            }
            boresight.push_back(q0);
            boresight.push_back(q1);
            boresight.push_back(q2);
            boresight.push_back(q3);
            hwpang.push_back(hwpoff);
            hwpoff += hwpdelta;
        }
    }

    std::map <std::string, toast::AlignedVector <int64_t> > checkindx;
    std::map <std::string, toast::AlignedVector <int64_t> > checkpix;
    std::map <std::string, toast::AlignedVector <double> > checkweight;

    bool gen_check = false;
    if (checkfile.compare("CREATE") == 0) {
        gen_check = true;
    } else {
        std::ifstream handle(checkfile);
        std::string line;
        while (std::getline(handle, line)) {
            std::istringstream iss(line);
            std::string dname;
            int64_t indx;
            int64_t pix;
            double w1;
            double w2;
            double w3;
            if (!(iss >> dname >> indx >> pix >> w1 >> w2 >> w3)) {
                break;
            }
            checkindx[dname].push_back(indx);
            checkpix[dname].push_back(pix);
            checkweight[dname].push_back(w1);
            checkweight[dname].push_back(w2);
            checkweight[dname].push_back(w3);
        }
    }

    gt.stop("Read inputs");

    gt.start("Total Calculation");

    int64_t nside = 2048;
    bool nest = true;

    std::map <std::string, toast::AlignedVector <int64_t> > detpixels;
    std::map <std::string, toast::AlignedVector <double> > detweights;

    for (auto const & dname : detnames) {
        detpixels[dname].clear();
        detweights[dname].clear();
        detpixels[dname].resize(hwpang.size());
        detweights[dname].resize(3 * hwpang.size());
    }

    #ifdef _OPENMP

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

    #else

    // Do something else...
    std::cerr << "Only OpenMP version is implemented!" << std::endl;
    exit(1);

    #endif

    gt.stop("Total Calculation");

    if (gen_check) {
        // Create the data
        std::ofstream handle("check.txt");
        size_t stride = 1000;
        for (auto const & dname : detnames) {
            size_t off = 0;
            while (off < hwpang.size()) {
                handle << std::setprecision(16)
                    << dname << " " << off << " " << detpixels[dname][off]
                    << " " << detweights[dname][3*off+0]
                    << " " << detweights[dname][3*off+1]
                    << " " << detweights[dname][3*off+2] << std::endl;
                off += stride;
            }
        }
    } else {
        // Compare data
        for (auto const & dname : detnames) {
            for (size_t i = 0; i < checkindx.size(); ++i) {
                auto indx = checkindx[dname][i];
                if (checkpix[dname][i] != detpixels[dname][indx]) {
                    std::cout << "Pixel mismatch:  detector " << dname
                        << ", sample " << indx << " " << detpixels[dname][indx]
                        << " != " << checkpix[dname][i] << std::endl;
                }
                auto const & cweight = checkweight.at(dname);
                double ci = cweight[3*i];
                double cq = cweight[3*i+1];
                double cu = cweight[3*i+2];
                auto const & dweight = detweights.at(dname);
                double di = dweight[3*indx];
                double dq = dweight[3*indx+1];
                double du = dweight[3*indx+2];
                double tol = 1.0e-5;

                if (fabs(ci) > tol) {
                    if (fabs((ci - di) / ci) > tol) {
                        std::cout << "Stokes I mismatch:  detector " << dname
                            << ", sample " << indx << " "
                            << di << " != " << ci << std::endl;
                    }
                } else if (fabs(ci - di) > tol) {
                    std::cout << "Stokes I mismatch:  detector " << dname
                        << ", sample " << indx << " "
                        << di << " != " << ci << std::endl;
                }

                if (fabs(cq) > tol) {
                    if (fabs((cq - dq) / cq) > tol) {
                        std::cout << "Stokes Q mismatch:  detector " << dname
                            << ", sample " << indx << " "
                            << dq << " != " << cq << std::endl;
                    }
                } else if (fabs(cq - dq) > tol) {
                    std::cout << "Stokes Q mismatch:  detector " << dname
                        << ", sample " << indx << " "
                        << dq << " != " << cq << std::endl;
                }

                if (fabs(cu) > tol) {
                    if (fabs((cu - du) / cu) > tol) {
                        std::cout << "Stokes U mismatch:  detector " << dname
                            << ", sample " << indx << " "
                            << du << " != " << cu << std::endl;
                    }
                } else if (fabs(cu - du) > tol) {
                    std::cout << "Stokes U mismatch:  detector " << dname
                        << ", sample " << indx << " "
                        << du << " != " << cu << std::endl;
                }

            }
        }
    }

    gt.report();

    return 0;
}
