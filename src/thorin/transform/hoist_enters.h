#ifndef THORIN_TRANSFORM_HOIST_ENTERS_H
#define THORIN_TRANSFORM_HOIST_ENTERS_H

#include "thorin/world.h"

namespace thorin {

class World;

void hoist_enters(std::unique_ptr<World>&);

}

#endif
