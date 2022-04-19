#include "thorin/analyses/looptree.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/cfg.h"

namespace thorin {

class World;

void structure_loops(World& world);
void structure_flow(World& world);

}