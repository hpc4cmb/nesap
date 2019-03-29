
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <utils.hpp>
#include <pointing_openmp.hpp>


int main(int argc, char * argv[]) {
    auto & gt = toast::GlobalTimers::get();

    // Read input data.

    if (argc != 3) {
        std::cerr << "Usage:  " << argv[0]
            << " <focalplane file> <boresight file>" << std::endl;
        return 1;
    }

    std::string fpfile(argv[1]);
    std::string borefile(argv[2]);

    std::vector <std::string> detnames;
    std::map <std::string, std::vector <double> > detquat;
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
            detquat[dname] = std::vector <double> (4);
            detquat[dname][0] = q0;
            detquat[dname][1] = q1;
            detquat[dname][2] = q2;
            detquat[dname][3] = q3;
            detcal[dname] = 1.0;
            deteps[dname] = 0.0;
        }
    }

    std::vector <double> boresight;
    std::vector <double> hwpang;

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

    int64_t nside = 2048;
    bool nest = true;

    std::map <std::string, std::vector <int64_t> > detpixels;
    std::map <std::string, std::vector <double> > detweights;

    for (auto const & dname : detnames) {
        detpixels[dname].clear();
        detweights[dname].clear();
    }

    toast::detector_pointing_healpix(nside, nest,
                                     boresight, hwpang,
                                     detnames, detquat,
                                     detcal, deteps,
                                     detpixels, detweights);

    return 0;
}
