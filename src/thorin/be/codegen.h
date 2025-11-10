#ifndef THORIN_CODEGEN_H
#define THORIN_CODEGEN_H

#include "thorin/world.h"

namespace thorin {

class CodeGen {
protected:
    CodeGen(World&, bool debug);
public:
    virtual ~CodeGen() {}

    virtual void emit_stream(std::ostream& stream) = 0;

    /// @name getters
    //@{
    World& world() const { return *world_; }
    bool debug() const { return debug_; }
    //@}

protected:
    std::unique_ptr<World> world_;
    bool debug_;
};

}

#endif
