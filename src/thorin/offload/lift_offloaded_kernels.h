#ifndef THORIN_LIFT_OFFLOADED_KERNELS_H
#define THORIN_LIFT_OFFLOADED_KERNELS_H

#include "offload.h"
#include "thorin/world.h"

namespace thorin {

void lower_offloaded_kernels(std::unique_ptr<World>&, Offload&);

}

#endif //THORIN_OFFLOAD_H
