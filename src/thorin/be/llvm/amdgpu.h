#ifndef THORIN_BE_LLVM_AMDGPU_H
#define THORIN_BE_LLVM_AMDGPU_H

#include "thorin/be/llvm/llvm.h"
#include "thorin/offload/kernel_config.h"

namespace thorin {

namespace llvm {

namespace llvm = ::llvm;

class AMDGPUCodeGen : public CodeGen {
public:
    AMDGPUCodeGen(World&, llvm::CallingConv::ID, llvm::CallingConv::ID, llvm::CallingConv::ID, const KernelConfigs&, int opt, bool debug);

protected:
    void emit_fun_decl_hook(Continuation*, llvm::Function*) override;
    llvm::Function* emit_fun_decl(Continuation*) override = 0;
    llvm::Value* emit_global(const Global*) override;
    llvm::Value* emit_mathop(llvm::IRBuilder<>&, const MathOp*) override;
    llvm::Value* emit_reserve(llvm::IRBuilder<>&, const Continuation*) override;
    std::string get_alloc_name() const override { return "malloc"; }

    const KernelConfigs& kernel_configs_;
};

}

}

#endif
