#include "offload.h"

#include "thorin/be/llvm/nvvm.h"
#include "thorin/be/llvm/amdgpu_hsa.h"
#include "thorin/be/llvm/amdgpu_pal.h"

namespace thorin {

struct AMDHSABackend : public Backend {
    explicit AMDHSABackend(Offload& b) : Backend(b) {
        b.register_intrinsic(Intrinsic::AMDGPUHSA, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        return std::make_unique<llvm::AMDGPUHSACodeGen>(*device_code_, kernel_configs_, backends_.thorin().opt(), backends_.thorin().debug());
    }

    std::string file_extension() override {
        return ".amdgpu";
    }
};

struct AMDPALBackend : public Backend {
    explicit AMDPALBackend(Offload& b) : Backend(b) {
        b.register_intrinsic(Intrinsic::AMDGPUPAL, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        return std::make_unique<llvm::AMDGPUPALCodeGen>(*device_code_, kernel_configs_, backends_.thorin().opt(), backends_.thorin().debug());
    }

    std::string file_extension() override {
        return ".amdgpu";
    }
};

struct NVVMBackend : public Backend {
    explicit NVVMBackend(Offload& b) : Backend(b) {
        b.register_intrinsic(Intrinsic::NVVM, *this);
    }

    std::unique_ptr<CodeGen> create_cg() override {
        return std::make_unique<llvm::NVVMCodeGen>(*device_code_, kernel_configs_, backends_.thorin().opt(), backends_.thorin().debug());
    }

    std::string file_extension() override {
        return ".nvvm";
    }
};

void register_llvm_offloading_backends(Offload& offload) {
    offload.register_backend(std::make_unique<AMDHSABackend>(offload));
    offload.register_backend(std::make_unique<AMDPALBackend>(offload));
    offload.register_backend(std::make_unique<NVVMBackend>(offload));
}

}
