#include "thorin.h"

// for colored output
#ifdef _WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

#include <cmath>

#if THORIN_ENABLE_CREATION_CONTEXT
#include <execinfo.h>
#endif

#if THORIN_ENABLE_RLIMITS
#include <sys/resource.h>
#endif

namespace thorin {

#ifdef COLORIZE_LOG
std::string World::colorize(const std::string& str, int color) {
    if (isatty(fileno(stdout))) {
        const char c = '0' + color;
        return "\033[1;3" + (c + ('m' + str)) + "\033[0m";
    }
#else
std::string World::colorize(const std::string& str, int) {
#endif
        return str;
    }

bool Thorin::ensure_stack_size(size_t new_size) {
#if THORIN_ENABLE_RLIMITS
    struct rlimit rl;
    int result = getrlimit(RLIMIT_STACK, &rl);
    if(result != 0) return false;

    rl.rlim_cur = new_size;
    result = setrlimit(RLIMIT_STACK, &rl);
    if(result != 0) return false;

    return true;
#else
    return false;
#endif
}

}