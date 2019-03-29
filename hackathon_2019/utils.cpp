
#include <utils.hpp>
#include <sstream>
#include <algorithm>
#include <cstring>


void * toast::aligned_alloc(size_t size, size_t align) {
    void * mem = NULL;
    int ret = posix_memalign(&mem, align, size);
    if (ret != 0) {
        std::ostringstream o;
        o << "cannot allocate " << size
          << " bytes of memory with alignment " << align;
        throw std::runtime_error(o.str().c_str());
    }
    memset(mem, 0, size);
    return mem;
}

void toast::aligned_free(void * ptr) {
    free(ptr);
    return;
}

toast::Timer::Timer() {
    clear();
}

void toast::Timer::start() {
    if (!running_) {
        start_ = std::chrono::high_resolution_clock::now();
        running_ = true;
        calls_++;
    }
    return;
}

void toast::Timer::stop() {
    if (running_) {
        stop_ = std::chrono::high_resolution_clock::now();
        std::chrono::duration <double> elapsed =
            std::chrono::duration_cast <std::chrono::duration <double> >
                (stop_ - start_);
        total_ += elapsed.count();
        running_ = false;
    }
    return;
}

void toast::Timer::clear() {
    start_ = time_point();
    stop_ = time_point();
    running_ = false;
    calls_ = 0;
    total_ = 0.0;
    return;
}

double toast::Timer::seconds() const {
    if (running_) {
        std::string msg("Timer is still running!");
        throw std::runtime_error(msg.c_str());
    }
    return total_;
}

bool toast::Timer::is_running() const {
    return running_;
}

void toast::Timer::report(char const * message) {
    double t = seconds();
    std::ostringstream msg;

    msg.precision(2);
    msg << std::fixed << message << ":  " << t << " seconds ("
        << calls_ << " calls)";
    std::cout << msg.str() << std::endl;
    return;
}

toast::GlobalTimers::GlobalTimers() {
    data.clear();
}

toast::GlobalTimers & toast::GlobalTimers::get() {
    static toast::GlobalTimers instance;

    return instance;
}

std::vector <std::string> toast::GlobalTimers::names() const {
    std::vector <std::string> ret;
    for (auto const & it : data) {
        ret.push_back(it.first);
    }
    std::stable_sort(ret.begin(), ret.end());
    return ret;
}

void toast::GlobalTimers::start(std::string const & name) {
    if (data.count(name) == 0) {
        data[name].clear();
    }
    data.at(name).start();
    return;
}

void toast::GlobalTimers::clear(std::string const & name) {
    data[name].clear();
    return;
}

void toast::GlobalTimers::stop(std::string const & name) {
    if (data.count(name) == 0) {
        std::ostringstream o;
        o << "Cannot stop timer " << name << " which does not exist";
        throw std::runtime_error(o.str().c_str());
    }
    data.at(name).stop();
    return;
}

double toast::GlobalTimers::seconds(std::string const & name) const {
    if (data.count(name) == 0) {
        std::ostringstream o;
        o << "Cannot get seconds for timer " << name
          << " which does not exist";
        throw std::runtime_error(o.str().c_str());
    }
    return data.at(name).seconds();
}

bool toast::GlobalTimers::is_running(std::string const & name) const {
    if (data.count(name) == 0) {
        return false;
    }
    return data.at(name).is_running();
}

void toast::GlobalTimers::stop_all() {
    for (auto & tm : data) {
        tm.second.stop();
    }
    return;
}

void toast::GlobalTimers::clear_all() {
    for (auto & tm : data) {
        tm.second.clear();
    }
    return;
}

void toast::GlobalTimers::report() {
    stop_all();
    std::vector <std::string> names;
    for (auto const & tm : data) {
        names.push_back(tm.first);
    }
    std::stable_sort(names.begin(), names.end());
    std::ostringstream msg;
    for (auto const & nm : names) {
        msg.str("");
        msg << "Global timer: " << nm;
        data.at(nm).report(msg.str().c_str());
    }
    return;
}
