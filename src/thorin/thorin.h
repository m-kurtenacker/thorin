#ifndef THORIN_H
#define THORIN_H

#include "thorin/world.h"

namespace thorin {

struct Offload;

/// Represents the complete compilation process
class Thorin {
public:
    explicit Thorin(const std::string& name, int opt, bool debug, std::string& hls_flags);

    Thorin(Thorin&) = delete;
    Thorin(const Thorin&&) = delete;

    int opt();
    bool debug();
    std::string& hls_flags();

    World& world() { return *world_; };
    std::unique_ptr<World>& world_container() { return world_; }

    /// Performs dead code, unreachable code and unused type elimination.
    void cleanup();
    /// Runs compilation pipeline according to flags
    void compile();

    /// Get offloading support. Must be accessed only after `compile` was ran!
    Offload& offload() { return *offload_; }

    bool ensure_stack_size(size_t new_size);

private:
    std::unique_ptr<World> world_;
    int opt_;
    bool debug_;
    std::string hls_flags_;
    std::unique_ptr<Offload> offload_;
};

void run_pass(std::unique_ptr<World>& world, std::function<void()> f, std::string pass_name);

#define RUN_PASS(w, pass) run_pass(w, [&]() { pass; }, #pass)

}

#endif
