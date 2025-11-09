#include "offload.h"

#include "thorin/be/c/c.h"

#include "thorin/transform/hls_channels.h"
#include "thorin/transform/hls_kernel_launch.h"
#include "thorin/transform/cleanup_world.h"

namespace thorin {

struct CudaBackend : public Backend {
    explicit CudaBackend(Offload& b) : Backend(b) {
        b.register_intrinsic(Intrinsic::CUDA, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        std::string empty;
        return std::make_unique<c::CodeGen>(*device_code_, kernel_configs_, c::Lang::CUDA, backends_.thorin().debug(), empty);
    }

    std::string file_extension() override {
        return ".cu";
    }
};

struct OpenCLBackend : public Backend {
    explicit OpenCLBackend(Offload& b) : Backend(b) {
        b.register_intrinsic(Intrinsic::OpenCL, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        std::string empty;
        return std::make_unique<c::CodeGen>(*device_code_, kernel_configs_, c::Lang::OpenCL, backends_.thorin().debug(), empty);
    }

    std::string file_extension() override {
        return ".cl";
    }
};

struct HLSBackend : public Backend {
    explicit HLSBackend(Offload& b) : Backend(b), hls_flags_(b.thorin().hls_flags()) {
        b.register_intrinsic(Intrinsic::HLS, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        Top2Kernel top2kernel;
        DeviceParams hls_host_params;

        hls_host_params = hls_channels(*device_code_, *importer_, top2kernel);
        cleanup_world(device_code_);
        hls_annotate_top(*device_code_, top2kernel, kernel_configs_);
        hls_kernel_launch(*device_code_, hls_host_params);

        return std::make_unique<c::CodeGen>(*device_code_, kernel_configs_, c::Lang::HLS, backends_.thorin().debug(), hls_flags_);
    }

    std::string file_extension() override {
        return ".hls";
    }

    std::string& hls_flags_;
};

void register_c_offloading_backends(Offload& offload) {
    offload.register_backend(std::make_unique<CudaBackend>(offload));
    offload.register_backend(std::make_unique<OpenCLBackend>(offload));
    offload.register_backend(std::make_unique<HLSBackend>(offload));
}

}
