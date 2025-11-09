#include "offload.h"
#include "lower_offload_intrinsics.h"

#include "thorin/transform/cleanup_world.h"

namespace thorin {

Backend::Backend(Offload& backends) : backends_(backends), device_code_(std::make_unique<World>(backends.thorin().world())) {}

void Backend::prepare_kernel_configs() {
    //device_code_.opt();
    cleanup_world(device_code_);

    Cont2Config adjusted_configs_map;

    auto conts = device_code_->copy_continuations();
    for (auto& [continuation, config] : kernel_configs_) {
        // recover the imported continuation (lost after the call to opt)
        Continuation* imported = nullptr;
        for (auto original_cont : conts) {
            if (!original_cont) continue;
            if (!original_cont->has_body()) continue;
            if (original_cont->name() == continuation->name())
                imported = original_cont;
        }
        assert(imported && "we lost a kernel ?");
        if (!imported) continue;

        adjusted_configs_map[imported] = std::move(config);
    }

    std::swap(kernel_configs_, adjusted_configs_map);
}

void Offload::register_backend(std::unique_ptr<Backend> backend) {
    backends_.push_back(std::move(backend));
}

void Offload::register_intrinsic(thorin::Intrinsic intrinsic, Backend& backend) {
    intrinsics_[intrinsic] = &backend;
}

Backend* Offload::find_backend_for_intrinsic(Intrinsic intrinsic) {
    auto handler = intrinsics_.find(intrinsic);
    if (handler == intrinsics_.end())
        return nullptr;
    return handler->second;
}

std::tuple<std::string, std::string> Offload::register_kernel_for_offloading(const App* launch, Continuation* kernel, std::unique_ptr<KernelConfig> config) {
    auto found = unique_kernel_.find(kernel);
    if (found != unique_kernel_.end())
        return found->second;
    auto backend = find_backend_for_intrinsic(launch->callee()->as_nom<Continuation>()->intrinsic());

    auto kernel_name = kernel->unique_name();
    auto filename = thorin().world().name() + backend->file_extension();
    auto r = std::make_tuple(filename, kernel_name);
    kernel->set_name(kernel_name);
    unique_kernel_[kernel] = r;

    // Import the continuation in the destination world
    Continuation* imported = backend->importer_->import(kernel)->as_nom<Continuation>();
    assert(imported);
    imported->world().make_external(imported);
    imported->attributes().cc = CC::C;

    // remove kernel from main world
    kernel->world().make_external(kernel);
    kernel->destroy("codegen");

    backend->kernel_configs_[kernel] = std::move(config);
    return r;
}

using GetProcessedFn = std::function<const Def*(const Def*)>;
using RebuildFn = std::function<const Def*(GetProcessedFn)>;
using ProcessMemberFn = std::function<void(const Def*, Defs, RebuildFn)>;
using ProcessBaseFn = std::function<void(const Def*)>;

void flatten(const Def* def, World& world, ProcessMemberFn member, ProcessBaseFn base) {
    auto t = def->type();

    if (auto tuple_t = t->isa<TupleType>()) {
        std::vector<const Def*> old_ops;
        for (size_t i = 0; i < tuple_t->num_ops(); i++)
            old_ops.push_back(world.extract(def, i));
        RebuildFn rebuild = [&](auto rebuild_op) -> const Def* {
            std::vector<const Def*> ops;
            for (auto old : old_ops)
                ops.push_back(rebuild_op(old));
            return world.tuple(ops);
        };
        member(def, old_ops, rebuild);
    } else if (auto struct_t = t->isa<StructType>()) {
        std::vector<const Def*> old_ops;
        for (size_t i = 0; i < struct_t->num_ops(); i++)
            old_ops.push_back(world.extract(def, i));
        RebuildFn rebuild = [&](auto rebuild_op) -> const Def* {
            std::vector<const Def*> ops;
            for (auto old : old_ops)
                ops.push_back(rebuild_op(old));
            return world.struct_agg(struct_t, ops);
        };
        member(def, old_ops, rebuild);
    } else {
        base(def);
    }
}

void default_lower_env_param(const Def* def, OffloadSite& context) {
    auto& world = context.rewriter->dst();
    ProcessMemberFn lower_member = [&](const Def* m, Defs ops, auto rebuild) -> void {
        // aggregate members are deconstructed and rebuilt for each kernel
        for (auto op : ops)
            default_lower_env_param(op, context);
        for (auto& k : context.kernels_) {
            k->insert_mapping(m, rebuild([&](const Def* old) { return k->get_mapping(old); }));
        }
    };

    auto lower_base = [&](const Def* old) -> void {
        auto fv = context.rewriter->instantiate(old);
        context.add_host_arg(fv);
        for (auto& k : context.kernels_) {
            k->insert_mapping(old, k->wrapper->append_param(fv->type()));
        }
    };
    flatten(def, world, lower_member, lower_base);
}

Offload::Offload(Thorin& thorin) : thorin_(thorin) {}

void Offload::offload(World& src) {
    for (auto& b : backends_) {
        b->importer_ = std::make_unique<Importer>(src, *b->device_code_);
    }

    lower_offload_intrinsics(thorin().world(), *this);

    for (auto& backend : backends_) {
        if (backend->world().empty())
            continue;

        backend->prepare_kernel_configs();
    }
}

}
