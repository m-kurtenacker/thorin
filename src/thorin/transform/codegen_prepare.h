#ifndef THORIN_TRANSFORM_CODEGEN_PREPARE_H
#define THORIN_TRANSFORM_CODEGEN_PREPARE_H

#include "thorin/world.h"

namespace thorin {

class World;

void codegen_prepare(std::unique_ptr<World>&);

}

#endif
