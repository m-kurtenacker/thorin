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

struct Backend {
    Backend(Offload&);
    virtual ~Backend() = default;

    virtual std::unique_ptr<CodeGen> create_cg() = 0;
    virtual std::string file_extension() = 0;

    World& world() { return *device_code_; }
    //Importer& importer() { return *importer_; }

    /*struct OffloadCallsite {
        const Def*& host_mem;
        virtual const Def* capture(const Def*);
        virtual const Def* add_host_arg(const Def*) = 0;

        struct OffloadKernel {
            Continuation* wrapper;
            const Def*& mem;
            void insert_mapping(const Def*, const Def*);
        };
        std::vector<OffloadKernel> kernels_;
    };

    virtual void lower_env_param(const Def* def, OffloadCallsite& context) {
        context.add_host_arg(def);

        for (auto& k : context.kernels_) {
            k.insert_mapping(def, k.wrapper->append_param(def->type()));
        }
    }*/

protected:
    Offload& backends_;
    std::unique_ptr<World> device_code_;
    std::unique_ptr<Importer> importer_;

    Cont2Config kernel_configs_;

    void prepare_kernel_configs();
    friend Offload;
};

}

#endif
