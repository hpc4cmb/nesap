
#ifndef SOLVER_OPENMP_HPP
#define SOLVER_OPENMP_HPP

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

void solver_lhs(
    int64_t nside, bool nest,
    toast::AlignedVector <double> const & boresight,
    toast::AlignedVector <double> const & hwpang,
    toast::AlignedVector <double> const & filter,
    toast::AlignedVector <std::string> const & detnames,
    std::map <std::string, toast::AlignedVector <double> > const & detquat,
    std::map <std::string, double> const & detcal,
    std::map <std::string, double> const & deteps, size_t nobs,
    toast::AlignedVector <double> & result);

}

#endif
