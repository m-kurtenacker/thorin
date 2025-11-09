#include "offload.h"

#include "thorin/be/spirv/spirv.h"
#include "thorin/transform/fungl_lower.h"

namespace thorin {

struct OpenCLSPIRVBackend : public Backend {
    explicit OpenCLSPIRVBackend(Offload& b) : Backend(b) {
        b.register_intrinsic(Intrinsic::OpenCL_SPIRV, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        spirv::Target target;
        return std::make_unique<spirv::CodeGen>(*device_code_, target, backends_.thorin().debug(), &kernel_configs_);
    }

    std::string file_extension() override {
        return ".cl.spv";
    }
};

struct LevelZeroSPIRVBackend : public Backend {
    explicit LevelZeroSPIRVBackend(Offload& b) : Backend(b) {
        b.register_intrinsic(Intrinsic::LevelZero_SPIRV, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        spirv::Target target;
        return std::make_unique<spirv::CodeGen>(*device_code_, target, backends_.thorin().debug(), &kernel_configs_);
    }

    std::string file_extension() override {
        return ".l0.spv";
    }
};

struct VulkanSPIRVBackend : public Backend {
    explicit VulkanSPIRVBackend(Offload& b) : Backend(b) {
        b.register_intrinsic(Intrinsic::VulkanCS_SPIRV, *this);
        b.register_intrinsic(Intrinsic::VulkanOffload_SPIRV, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        spirv::Target target;
        target.bugs = {};
        target.dialect = spirv::Target::Vulkan;
        fungl_lower(device_code_, true);
        return std::make_unique<spirv::CodeGen>(*device_code_, target, backends_.thorin().debug(), &kernel_configs_);
    }

    std::string file_extension() override {
        return ".vk.spv";
    }
};

void register_spirv_offloading_backends(Offload& offload) {
    offload.register_backend(std::make_unique<OpenCLSPIRVBackend>(offload));
    offload.register_backend(std::make_unique<LevelZeroSPIRVBackend>(offload));
    offload.register_backend(std::make_unique<VulkanSPIRVBackend>(offload));
}

}
