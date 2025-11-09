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

/*
* optimizations
 */
void Thorin::compile() {
    bool debug_passes = getenv("THORIN_DEBUG_PASSES");
#define RUN_PASS(pass) \
{ \
world().VLOG("running pass {}", #pass);  \
pass;                                    \
debug_verify(world());                   \
if (debug_passes) world().dump_scoped(); \
}

    RUN_PASS(cleanup())
    RUN_PASS(flatten_tuples(*this))
    RUN_PASS(split_slots(*this))
    RUN_PASS(lift(*this));
    //RUN_PASS(inliner(*this))
    RUN_PASS(hoist_enters(*this))
    RUN_PASS(dead_load_opt(world()))
    RUN_PASS(lower_closure_env(*this));
    //RUN_PASS(cleanup())
    RUN_PASS(codegen_prepare(*this))
    offload_->offload(world());
    fungl_lower(world_container(), false);
}

void Thorin::cleanup() { cleanup_world(world_container()); }

}
