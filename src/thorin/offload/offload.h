#ifndef THORIN_OFFLOAD_H
#define THORIN_OFFLOAD_H

#include "kernel_config.h"

#include "thorin/thorin.h"
#include "thorin/be/codegen.h"
#include "thorin/transform/importer.h"

namespace thorin {

struct Backend;

struct Offload {
    Offload(Thorin&);
    Offload(Offload&) = delete;

    Thorin& thorin() { return thorin_; }

    const std::vector<std::unique_ptr<Backend>>& backends() {
        return backends_;
    }

    void register_backend(std::unique_ptr<Backend>);
    void register_intrinsic(Intrinsic, Backend&);

    Backend* find_backend_for_intrinsic(Intrinsic);

    std::tuple<std::string, std::string> register_kernel_for_offloading(const App* launch, Continuation*, std::unique_ptr<KernelConfig>);
private:
    Thorin& thorin_;
    std::vector<std::unique_ptr<Backend>> backends_;
    std::unordered_map<Intrinsic, Backend*> intrinsics_;
    ContinuationMap<std::tuple<std::string, std::string>> unique_kernel_;

    void offload(World&);

    friend Thorin;
    friend Backend;
};

/// Represents a callsite where an offload intrinsic is called, used when lifting kernel free variables
struct OffloadSite {
    Rewriter* rewriter;
    virtual const Def*& host_mem() = 0;
    OffloadSite(Rewriter* rewriter) : rewriter(rewriter) {}

    virtual void add_host_arg(const Def*) = 0;

    /// Note: an offload site might have more than one kernel!
    /// See VulkanOffload
    struct Kernel {
        Continuation* old_kernel;
        Continuation* wrapper;
        virtual const Def*& mem() = 0;

        virtual void insert_mapping(const Def*, const Def*) = 0;
        virtual const Def* get_mapping(const Def*) = 0;

        Kernel(Continuation* o, Continuation* w) : old_kernel(o), wrapper(w) {}
    };
    std::vector<std::unique_ptr<Kernel>> kernels_;
};

/// Simple strategy that deconstructs aggregates and adds params to the launch side
void default_lower_env_param(const Def* def, OffloadSite& context);

struct Backend {
    Backend(Offload&);
    virtual ~Backend() = default;

    virtual std::unique_ptr<CodeGen> create_cg() = 0;
    virtual std::string file_extension() = 0;

    World& world() { return *device_code_; }

    virtual void lower_env_param(const Def* def, OffloadSite& context) {
        return default_lower_env_param(def, context);
    }

protected:
    Offload& backends_;
    std::unique_ptr<World> device_code_;
    std::unique_ptr<Importer> importer_;

    KernelConfigs kernel_configs_;

    void prepare_kernel_configs();
    friend Offload;
};

void register_c_offloading_backends(Offload&);

#if THORIN_ENABLE_SPIRV
void register_spirv_offloading_backends(Offload&);
#endif

#if THORIN_ENABLE_LLVM
void register_llvm_offloading_backends(Offload&);
#endif

}

#endif
