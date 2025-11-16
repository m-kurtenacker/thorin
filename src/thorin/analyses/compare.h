#ifndef THORIN_ANALYSES_COMPARE_H
#define THORIN_ANALYSES_COMPARE_H

#include "thorin/world.h"

namespace thorin {

bool compare_defs(const Def* a, const Def* b);
bool compare_worlds(World& a, World& b);

}

#endif
