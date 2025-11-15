#include "thorin/world.h"
#include "thorin/analyses/scope.h"

namespace thorin {

DefSet spillable_free_defs(Continuation* entry, ScopesForest& forest, DefSet& result, DefSet& rematerialize);

void lift(std::unique_ptr<World>&);

}
