#ifndef THORIN_H
#define THORIN_H

#include "thorin/world.h"

namespace thorin {

class Thorin {
public:
    /// Initial world constructor
    explicit Thorin(const std::string& name);
    explicit Thorin(World& src);

    Thorin(Thorin&) = delete;
    Thorin(const Thorin&&) = delete;

    World& world() { return *world_; };
    std::unique_ptr<World>& world_container() { return world_; }

    /// Performs dead code, unreachable code and unused type elimination.
    void cleanup();
    void opt();

    bool ensure_stack_size(size_t new_size);

private:
    std::unique_ptr<World> world_;
};

}

#endif
