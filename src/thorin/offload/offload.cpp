#include "offload.h"
#include "runtime.h"
#include "lower_offload_intrinsics.h"

#include "thorin/be/c/c.h"

#if THORIN_ENABLE_LLVM
#include "thorin/be/llvm/nvvm.h"
#include "thorin/be/llvm/amdgpu_hsa.h"
#include "thorin/be/llvm/amdgpu_pal.h"
#endif

#if THORIN_ENABLE_SPIRV
#include "thorin/be/spirv/spirv.h"
#endif

#include "thorin/transform/hls_channels.h"
#include "thorin/transform/hls_kernel_launch.h"
#include "thorin/transform/cleanup_world.h"

namespace thorin {

Backend::Backend(thorin::DeviceBackends& backends, World& src) : backends_(backends), device_code_(std::make_unique<World>(src)), importer_(std::make_unique<Importer>(src, *device_code_)) {}

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

void DeviceBackends::register_backend(std::unique_ptr<Backend> backend) {
    backends_.push_back(std::move(backend));
}

World& DeviceBackends::world() { return world_; }
bool DeviceBackends::debug() { return debug_; }
int DeviceBackends::opt() { return opt_; }

void DeviceBackends::register_intrinsic(thorin::Intrinsic intrinsic, Backend& backend) {
    intrinsics_[intrinsic] = &backend;
}

std::tuple<std::string, std::string> DeviceBackends::register_kernel_for_offloading(const App* launch, Continuation* kernel, std::unique_ptr<KernelConfig> config) {
    auto found = unique_kernel_.find(kernel);
    if (found != unique_kernel_.end())
        return found->second;
    Continuation* intrinsic_cont = launch->callee()->as_nom<Continuation>();
    auto handler = intrinsics_.find(intrinsic_cont->intrinsic());
    assert(handler != intrinsics_.end());
    auto backend = handler->second;

    auto kernel_name = kernel->unique_name();
    auto filename = world().name() + backend->file_extension();
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

struct CudaBackend : public Backend {
    explicit CudaBackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::CUDA, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        std::string empty;
        return std::make_unique<c::CodeGen>(*device_code_, kernel_configs_, c::Lang::CUDA, backends_.debug(), empty);
    }

    std::string file_extension() override {
        return ".cu";
    }
};

struct OpenCLBackend : public Backend {
    explicit OpenCLBackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::OpenCL, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        std::string empty;
        return std::make_unique<c::CodeGen>(*device_code_, kernel_configs_, c::Lang::OpenCL, backends_.debug(), empty);
    }

    std::string file_extension() override {
        return ".cl";
    }
};

#if THORIN_ENABLE_SPIRV
struct OpenCLSPIRVBackend : public Backend {
    explicit OpenCLSPIRVBackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::OpenCL_SPIRV, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        spirv::Target target;
        return std::make_unique<spirv::CodeGen>(*device_code_, target, backends_.debug(), &kernel_configs_);
    }

    std::string file_extension() override {
        return ".cl.spv";
    }
};

struct LevelZeroSPIRVBackend : public Backend {
    explicit LevelZeroSPIRVBackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::LevelZero_SPIRV, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        spirv::Target target;
        return std::make_unique<spirv::CodeGen>(*device_code_, target, backends_.debug(), &kernel_configs_);
    }

    std::string file_extension() override {
        return ".l0.spv";
    }
};

struct VulkanSPIRVBackend : public Backend {
    explicit VulkanSPIRVBackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::VulkanCS_SPIRV, *this);
        b.register_intrinsic(Intrinsic::VulkanOffload_SPIRV, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        spirv::Target target;
        target.bugs = {};
        target.dialect = spirv::Target::Vulkan;
        return std::make_unique<spirv::CodeGen>(*device_code_, target, backends_.debug(), &kernel_configs_);
    }

    std::string file_extension() override {
        return ".vk.spv";
    }
};
#endif

#if THORIN_ENABLE_LLVM
struct AMDHSABackend : public Backend {
    explicit AMDHSABackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::AMDGPUHSA, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        return std::make_unique<llvm::AMDGPUHSACodeGen>(*device_code_, kernel_configs_, backends_.opt(), backends_.debug());
    }

    std::string file_extension() override {
        return ".amdgpu";
    }
};

struct AMDPALBackend : public Backend {
    explicit AMDPALBackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::AMDGPUPAL, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        return std::make_unique<llvm::AMDGPUPALCodeGen>(*device_code_, kernel_configs_, backends_.opt(), backends_.debug());
    }

    std::string file_extension() override {
        return ".amdgpu";
    }
};

struct NVVMBackend : public Backend {
    explicit NVVMBackend(DeviceBackends& b, World& src) : Backend(b, src) {
        b.register_intrinsic(Intrinsic::NVVM, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        return std::make_unique<llvm::NVVMCodeGen>(*device_code_, kernel_configs_, backends_.opt(), backends_.debug());
    }

    std::string file_extension() override {
        return ".nvvm";
    }
};
#endif

struct HLSBackend : public Backend {
    explicit HLSBackend(DeviceBackends& b, World& src, std::string& hls_flags) : Backend(b, src), hls_flags_(hls_flags) {
        b.register_intrinsic(Intrinsic::HLS, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        Top2Kernel top2kernel;
        DeviceParams hls_host_params;

        hls_host_params = hls_channels(*device_code_, *importer_, top2kernel);
        cleanup_world(device_code_);
        hls_annotate_top(*device_code_, top2kernel, kernel_configs_);
        hls_kernel_launch(*device_code_, hls_host_params);

        return std::make_unique<c::CodeGen>(*device_code_, kernel_configs_, c::Lang::HLS, backends_.debug(), hls_flags_);
    }

    std::string file_extension() override {
        return ".hls";
    }

    std::string& hls_flags_;
};

DeviceBackends::DeviceBackends(World& world, int opt, bool debug, std::string& hls_flags) : world_(world), opt_(opt), debug_(debug) {
    register_backend(std::make_unique<CudaBackend>(*this, world_));
    register_backend(std::make_unique<OpenCLBackend>(*this, world_));
#if THORIN_ENABLE_LLVM
    register_backend(std::make_unique<AMDHSABackend>(*this, world_));
    register_backend(std::make_unique<AMDPALBackend>(*this, world_));
    register_backend(std::make_unique<NVVMBackend>(*this, world_));
#endif
#if THORIN_ENABLE_SPIRV
    register_backend(std::make_unique<OpenCLSPIRVBackend>(*this, world_));
    register_backend(std::make_unique<LevelZeroSPIRVBackend>(*this, world_));
    register_backend(std::make_unique<VulkanSPIRVBackend>(*this, world_));
#endif
    register_backend(std::make_unique<HLSBackend>(*this, world_, hls_flags));

    lower_offload_intrinsics(world, *this);

    for (auto& backend : backends_) {
        if (backend->world().empty())
            continue;

        backend->prepare_kernel_configs();
    }
}

}
