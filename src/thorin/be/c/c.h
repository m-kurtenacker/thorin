#ifndef THORIN_BE_C_H
#define THORIN_BE_C_H

#include <cstdint>
#include <iostream>

#include "thorin/be/codegen.h"
#include "thorin/thorin.h"

namespace thorin {

class World;

namespace c {

enum class Lang : uint8_t { C99, HLS, CUDA, OpenCL };

class CodeGen : public thorin::CodeGen {
public:
    CodeGen(World& world, const KernelConfigs& kernel_configs, Lang lang, bool debug, std::string& flags);

    void emit_stream(std::ostream& stream) override;

private:
    const KernelConfigs& kernel_configs_;
    Lang lang_;
    bool debug_;
    std::string flags_;
};

void emit_c_int(Thorin&, Stream& stream);

}

}

#endif
