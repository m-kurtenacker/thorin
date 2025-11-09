#ifndef FUNGL_TRANSFORM_LOWER
#define FUNGL_TRANSFORM_LOWER

#include "thorin/world.h"

namespace thorin {

void fungl_lower(std::unique_ptr<World>& world, bool is_device);

}

#endif
