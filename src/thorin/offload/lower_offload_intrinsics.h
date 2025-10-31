#ifndef THORIN_OFFLOAD_INTRINSICS_H
#define THORIN_OFFLOAD_INTRINSICS_H

#include "offload.h"
#include "thorin/world.h"

namespace thorin {

enum class KernelArgType : uint8_t { Val = 0, Ptr, Struct };

void lower_offload_intrinsics(World&, Offload&);

}

#endif //THORIN_OFFLOAD_H
