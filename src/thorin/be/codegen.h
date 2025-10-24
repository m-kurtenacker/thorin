#ifndef THORIN_CODEGEN_H
#define THORIN_CODEGEN_H

#include "thorin/transform/importer.h"
#include "thorin/be/kernel_config.h"

namespace thorin {

class CodeGen {
protected:
    CodeGen(Thorin& thorin, bool debug);
public:
    virtual ~CodeGen() {}

    virtual void emit_stream(std::ostream& stream) = 0;

    /// @name getters
    //@{
    Thorin& thorin() const { return thorin_; }
    World& world() const { return thorin().world(); }
    bool debug() const { return debug_; }
    //@}

private:
    Thorin& thorin_;
    bool debug_;
};

struct DeviceBackends;

struct Backend {
    Backend(DeviceBackends& backends, World& src);
    virtual ~Backend() = default;

    virtual std::unique_ptr<CodeGen> create_cg() = 0;
    virtual std::string file_extension() = 0;

    Thorin& thorin() { return device_code_; }
    Importer& importer() { return *importer_; }

protected:
    DeviceBackends& backends_;
    Thorin device_code_;
    std::unique_ptr<Importer> importer_;

    Cont2Config kernel_configs_;

    void prepare_kernel_configs();
    friend DeviceBackends;
};

using GetKernelConfigFn = std::unique_ptr<KernelConfig>(const App*, Continuation*);
GetKernelConfigFn get_hls_kernel_config;
GetKernelConfigFn get_compute_kernel_config;

struct DeviceBackends {
    DeviceBackends(World& world, int opt, bool debug, std::string& hls_flags);

    DeviceBackends(DeviceBackends&) = delete;

    World& world();
    const std::vector<std::unique_ptr<Backend>>& backends() {
        return backends_;
    }

    int opt();
    bool debug();

    void register_backend(std::unique_ptr<Backend>);
    void register_intrinsic(Intrinsic, Backend&);

    std::tuple<std::string, std::string> register_kernel_for_offloading(const App* launch, Continuation*, std::unique_ptr<KernelConfig>);
private:
    World& world_;
    std::vector<std::unique_ptr<Backend>> backends_;
    std::unordered_map<Intrinsic, Backend*> intrinsics_;
    ContinuationMap<std::tuple<std::string, std::string>> unique_kernel_;

    int opt_;
    bool debug_;
friend Backend;
};

}

#endif
