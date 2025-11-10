#include "thorin/be/codegen.h"
#include "thorin/transform/importer.h"

namespace thorin {

CodeGen::CodeGen(World& world, bool debug)
    : world_(std::make_unique<World>(world))
    , debug_(debug) {
    import_world(*world_ ,world);
}

}
