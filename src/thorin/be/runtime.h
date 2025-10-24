#ifndef THORIN_RUNTIME_H
#define THORIN_RUNTIME_H

/// Backend-agnostic information to interface with the runtime component
namespace thorin {

enum Platform {
    CPU_PLATFORM = 0,
    CUDA_PLATFORM = 1,
    OPENCL_PLATFORM = 2,
    HSA_PLATFORM = 3,
    PAL_PLATFORM = 4,
    LEVEL_ZERO_PLATFORM = 5,
    VULKAN_PLATFORM = 6,
};

enum KernelLaunchArgs {
    Mem = 0,
    Device,
    Space,
    Config,
    Body,
    Return,
    Num
};

enum class ParallelForArgs {
    Mem = 0,
    NumThreads,
    Lower,
    Upper,
    Fun,
    Return,
    Num
};

enum class SpawnFibersArgs {
    Mem = 0,
    NumThreads,
    NumBlocks,
    NumWarps,
    Fun,
    Return,
    Num
};

enum class SpawnThreadArgs {
    Mem = 0,
    Fun,
    Return,
    Num
};

enum class SyncArgs {
    Mem = 0,
    Id,
    Return,
};

}

#endif
