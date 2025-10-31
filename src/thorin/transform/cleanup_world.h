#ifndef THORIN_TRANSFORM_CLEAN_WORLD
#define THORIN_TRANSFORM_CLEAN_WORLD

#include "thorin/world.h"

namespace thorin {

void cleanup_world(std::unique_ptr<World>& world);

}

#endif
