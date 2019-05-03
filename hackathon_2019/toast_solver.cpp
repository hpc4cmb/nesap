
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>

#include <utils.hpp>

#ifdef USE_OPENMP
#include <solver_openmp.hpp>
#else
#include <solver_cuda.hpp>
#endif


int main(int argc, char * argv[]) {
    auto & gt = toast::GlobalTimers::get();

    // Number of observations
    size_t nobs = 90;
    //size_t nobs = 1;

    // Read input data.

    if (argc != 5) {
        std::cerr << "Usage:  " << argv[0]
            << " <focalplane file> <boresight file> <filter file> <check file>" << std::endl;
        return 1;
    }

    gt.start("Read inputs");

    std::string fpfile(argv[1]);
    std::string borefile(argv[2]);
    std::string filtfile(argv[3]);
    std::string checkfile(argv[4]);

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

    toast::AlignedVector <double> filter;
    {
        std::ifstream handle(filtfile);
        std::string line;
        while (std::getline(handle, line)) {
            std::istringstream iss(line);
            double psd;
            if (!(iss >> psd)) {
                break;
            }
            filter.push_back(psd);
        }
    }

    // toast::AlignedVector <double> checkmap;
    //
    // bool gen_check = false;
    // if (checkfile.compare("CREATE") == 0) {
    //     gen_check = true;
    // } else {
    //     std::ifstream handle(checkfile);
    //     std::string line;
    //     while (std::getline(handle, line)) {
    //         std::istringstream iss(line);
    //         double mval;
    //         if (!(iss >> mval)) {
    //             break;
    //         }
    //         checkmap.push_back(mval);
    //     }
    // }

    gt.stop("Read inputs");

    gt.start("Total Calculation");

    int64_t nside = 2048;
    bool nest = true;

    toast::AlignedVector <double> result;

    toast::solver_lhs(nside, nest, boresight, hwpang, filter, detnames,
        detquat, detcal, deteps, nobs, result);

    gt.stop("Total Calculation");

    // if (gen_check) {
    //     // Create the data
    //     std::ofstream handle("check.txt");
    //     size_t stride = 1000;
    //     for (auto const & dname : detnames) {
    //         size_t off = 0;
    //         while (off < hwpang.size()) {
    //             handle << std::setprecision(16)
    //                 << dname << " " << off << " " << detpixels[dname][off]
    //                 << " " << detweights[dname][3*off+0]
    //                 << " " << detweights[dname][3*off+1]
    //                 << " " << detweights[dname][3*off+2] << std::endl;
    //             off += stride;
    //         }
    //     }
    // } else {
    //     // Compare data
    //     for (auto const & dname : detnames) {
    //         for (size_t i = 0; i < checkindx.size(); ++i) {
    //             auto indx = checkindx[dname][i];
    //             if (checkpix[dname][i] != detpixels[dname][indx]) {
    //                 std::cout << "Pixel mismatch:  detector " << dname
    //                     << ", sample " << indx << " " << detpixels[dname][indx]
    //                     << " != " << checkpix[dname][i] << std::endl;
    //             }
    //             auto const & cweight = checkweight.at(dname);
    //             double ci = cweight[3*i];
    //             double cq = cweight[3*i+1];
    //             double cu = cweight[3*i+2];
    //             auto const & dweight = detweights.at(dname);
    //             double di = dweight[3*indx];
    //             double dq = dweight[3*indx+1];
    //             double du = dweight[3*indx+2];
    //             double tol = 1.0e-5;
    //
    //             if (fabs(ci) > tol) {
    //                 if (fabs((ci - di) / ci) > tol) {
    //                     std::cout << "Stokes I mismatch:  detector " << dname
    //                         << ", sample " << indx << " "
    //                         << di << " != " << ci << std::endl;
    //                 }
    //             } else if (fabs(ci - di) > tol) {
    //                 std::cout << "Stokes I mismatch:  detector " << dname
    //                     << ", sample " << indx << " "
    //                     << di << " != " << ci << std::endl;
    //             }
    //
    //             if (fabs(cq) > tol) {
    //                 if (fabs((cq - dq) / cq) > tol) {
    //                     std::cout << "Stokes Q mismatch:  detector " << dname
    //                         << ", sample " << indx << " "
    //                         << dq << " != " << cq << std::endl;
    //                 }
    //             } else if (fabs(cq - dq) > tol) {
    //                 std::cout << "Stokes Q mismatch:  detector " << dname
    //                     << ", sample " << indx << " "
    //                     << dq << " != " << cq << std::endl;
    //             }
    //
    //             if (fabs(cu) > tol) {
    //                 if (fabs((cu - du) / cu) > tol) {
    //                     std::cout << "Stokes U mismatch:  detector " << dname
    //                         << ", sample " << indx << " "
    //                         << du << " != " << cu << std::endl;
    //                 }
    //             } else if (fabs(cu - du) > tol) {
    //                 std::cout << "Stokes U mismatch:  detector " << dname
    //                     << ", sample " << indx << " "
    //                     << du << " != " << cu << std::endl;
    //             }
    //
    //         }
    //     }
    // }

    gt.report();

    return 0;
}
