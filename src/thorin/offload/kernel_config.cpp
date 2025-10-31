#include "kernel_config.h"
#include "runtime.h"

#include "thorin/transform/hls_kernel_launch.h"

namespace thorin {

static const App* get_alloc_call(const Def* def) {
    // look through casts
    while (auto conv_op = def->isa<ConvOp>())
        def = conv_op->op(0);

    auto param = def->isa<Param>();
    if (!param) return nullptr;

    auto ret = param->continuation();
    for (auto use : ret->uses()) {
        auto call = use.def()->isa<App>();
        if (!call || use.index() == 0) continue;

        auto callee = call->callee();
        if (callee->name() != "anydsl_alloc") continue;

        return call;
    }
    return nullptr;
}

std::unique_ptr<KernelConfig> get_compute_kernel_config(const App* app, Continuation* /* imported */) {
    // determine whether or not this kernel uses restrict pointers
    bool has_restrict = true;
    DefSet allocs;
    for (size_t i = KernelLaunchArgs::Num, e = app->num_args(); has_restrict && i != e; ++i) {
        auto arg = app->arg(i);
        if (!arg->type()->isa<PtrType>()) continue;
        auto alloc = get_alloc_call(arg);
        if (!alloc) has_restrict = false;
        auto p = allocs.insert(alloc);
        has_restrict &= p.second;
    }

    auto it_config = app->arg(KernelLaunchArgs::Config)->isa<Tuple>();
    if (it_config &&
        it_config->op(0)->isa<PrimLit>() &&
        it_config->op(1)->isa<PrimLit>() &&
        it_config->op(2)->isa<PrimLit>()) {
        return std::make_unique<GPUKernelConfig>(std::tuple<int, int, int>{
                it_config->op(0)->as<PrimLit>()->qu32_value().data(),
                it_config->op(1)->as<PrimLit>()->qu32_value().data(),
                it_config->op(2)->as<PrimLit>()->qu32_value().data()
        }, has_restrict);
        }
    return std::make_unique<GPUKernelConfig>(std::tuple<int, int, int>{-1, -1, -1}, has_restrict);
}

static uint64_t get_alloc_size(const Def* def) {
    auto call = get_alloc_call(def);
    if (!call) return 0;

    // signature: anydsl_alloc(mem, i32, i64, fn(mem, &[i8]))
    auto size = call->arg(2)->isa<PrimLit>();
    return size ? static_cast<uint64_t>(size->value().get_qu64()) : 0_u64;
}

std::unique_ptr<KernelConfig> get_hls_kernel_config(const App* app, Continuation* kernel) {
    World& world = app->world();
    HLSKernelConfig::Param2Size param_sizes;
    for (size_t i = hls_free_vars_offset, e = app->num_args(); i != e; ++i) {
        auto arg = app->arg(i);
        auto ptr_type = arg->type()->isa<PtrType>();
        if (!ptr_type) continue;
        auto size = get_alloc_size(arg);
        if (size == 0)
            world.edef(arg, "array size is not known at compile time");
        auto elem_type = ptr_type->pointee();
        size_t multiplier = 1;
        if (!elem_type->isa<PrimType>()) {
            if (auto array_type = elem_type->isa<ArrayType>())
                elem_type = array_type->elem_type();
        }
        if (!elem_type->isa<PrimType>()) {
            if (auto def_array_type = elem_type->isa<DefiniteArrayType>()) {
                elem_type = def_array_type->elem_type();
                multiplier = def_array_type->dim();
            }
        }
        auto prim_type = elem_type->isa<PrimType>();
        if (!prim_type)
            world.edef(arg, "only pointers to arrays of primitive types are supported");
        auto num_elems = size / (multiplier * num_bits(prim_type->primtype_tag()) / 8);
        // imported has type: fn (mem, fn (mem), ...)
        param_sizes.emplace(kernel->param(i - hls_free_vars_offset + 2), num_elems);
    }
    return std::make_unique<HLSKernelConfig>(param_sizes);
}

}
