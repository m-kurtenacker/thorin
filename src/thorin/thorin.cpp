#include "thorin.h"

#include "thorin/analyses/verify.h"
#include "thorin/offload/offload.h"

#include "thorin/transform/cleanup_world.h"
#include "thorin/transform/codegen_prepare.h"
#include "thorin/transform/dead_load_opt.h"
#include "thorin/transform/flatten_tuples.h"
#include "thorin/transform/hoist_enters.h"
#include "thorin/transform/inliner.h"
#include "thorin/transform/lift.h"
#include "thorin/transform/lower_closure_env.h"
#include "thorin/transform/split_slots.h"
#include "thorin/transform/fungl_lower.h"

#include "thorin/offload/lift_offloaded_kernels.h"

namespace thorin {

Thorin::Thorin(const std::string& name, int opt, bool debug, std::string& hls_flags)
    : world_(std::make_unique<World>(name))
    , opt_(opt)
    , debug_(debug)
    , hls_flags_(hls_flags)
    , offload_(std::make_unique<Offload>(*this))
{}

bool Thorin::debug() { return debug_; }
int Thorin::opt() { return opt_; }
std::string& Thorin::hls_flags() { return hls_flags_; }

void run_pass(std::unique_ptr<World>& world, std::function<void()> f, std::string pass_name) {
    world->VLOG("running pass {}", pass_name);
    f();
    debug_verify(*world);
    bool debug_passes = getenv("THORIN_DEBUG_PASSES");
    if (debug_passes)
        world->dump_scoped();
}

/*
* optimizations
 */
void Thorin::compile() {
    RUN_PASS(world_container(), cleanup());
    RUN_PASS(world_container(), flatten_tuples(world_container()));
    RUN_PASS(world_container(), split_slots(world_container()));
    RUN_PASS(world_container(), lower_offloaded_kernels(world_container(), offload()));
    RUN_PASS(world_container(), lift(world_container()));
    //RUN_PASS(inliner(*this))
    RUN_PASS(world_container(), hoist_enters(world_container()));
    RUN_PASS(world_container(), dead_load_opt(world()));
    RUN_PASS(world_container(), offload_->offload(world()));
    RUN_PASS(world_container(), fungl_lower(world_container(), false));
}

void Thorin::cleanup() { cleanup_world(world_container()); }

}
