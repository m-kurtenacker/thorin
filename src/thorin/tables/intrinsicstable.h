#ifndef THORIN_INTRINSIC
#error "define THORIN_INTRINSIC before including this file"
#endif
#ifndef THORIN_OFFLOAD
#define THORIN_OFFLOAD(X, Y) THORIN_ACCELERATOR(X, Y)
#endif
#ifndef THORIN_ACCELERATOR
#define THORIN_ACCELERATOR(X, Y) THORIN_INTRINSIC(X, Y)
#endif

THORIN_OFFLOAD(CUDA, "cuda")                          ///< Internal CUDA-Backend.
THORIN_OFFLOAD(NVVM, "nvvm")                          ///< Internal NNVM-Backend.
THORIN_OFFLOAD(OpenCL, "opencl")                      ///< Internal OpenCL-Backend.
THORIN_OFFLOAD(OpenCL_SPIRV, "opencl_spirv")          ///< Internal OpenCL-Backend.
THORIN_OFFLOAD(LevelZero_SPIRV, "levelzero")          ///< Internal SPIRV for Level0-Backend.
THORIN_OFFLOAD(VulkanCS_SPIRV, "vulkan_cs")           ///< Internal SPIRV for Vulkan-Backend.
THORIN_OFFLOAD(VulkanOffload_SPIRV, "vulkan_offload") ///< Internal SPIRV for Vulkan-Backend.
THORIN_OFFLOAD(AMDGPUHSA, "amdgpu_hsa")               ///< Internal AMDGPU-HSA-Backend.
THORIN_OFFLOAD(AMDGPUPAL, "amdgpu_pal")               ///< Internal AMDGPU-PAL-Backend.
THORIN_OFFLOAD(HLS, "hls")                            ///< Internal HLS-Backend.

THORIN_ACCELERATOR(Parallel, "parallel")   ///< Internal Parallel-CPU-Backend.
THORIN_ACCELERATOR(Fibers, "fibers")       ///< Internal Parallel-CPU-Backend using resumable fibers.
THORIN_ACCELERATOR(Spawn, "spawn")         ///< Internal Parallel-CPU-Backend.
THORIN_ACCELERATOR(Sync, "sync")           ///< Internal Parallel-CPU-Backend.
THORIN_ACCELERATOR(Vectorize, "vectorize") ///< External vectorizer.

THORIN_INTRINSIC(Reserve, "reserve_shared")             ///< Intrinsic memory reserve function
THORIN_INTRINSIC(Atomic, "atomic")                      ///< Intrinsic atomic function
THORIN_INTRINSIC(AtomicLoad, "atomic_load")             ///< Intrinsic atomic load function
THORIN_INTRINSIC(AtomicStore, "atomic_store")           ///< Intrinsic atomic store function
THORIN_INTRINSIC(CmpXchg, "cmpxchg")                    ///< Intrinsic cmpxchg function
THORIN_INTRINSIC(CmpXchgWeak, "cmpxchg_weak")           ///< Intrinsic cmpxchg weak function
THORIN_INTRINSIC(Fence, "fence")                        ///< Intrinsic fence function
THORIN_INTRINSIC(Undef, "undef")                        ///< Intrinsic undef function
THORIN_INTRINSIC(PipelineContinue, "pipeline_continue") ///< Intrinsic loop-pipelining-HLS-Backend
THORIN_INTRINSIC(Pipeline, "pipeline")                  ///< Intrinsic loop-pipelining-HLS-Backend
THORIN_INTRINSIC(Branch, "branch")                      ///< branch(mem, cond, T, F).
THORIN_INTRINSIC(Match, "match")                        ///< match(mem, val, otherwise, (case1, cont1), (case2, cont2), ...)
THORIN_INTRINSIC(PeInfo, "pe_info")                     ///< Partial evaluation debug info.
THORIN_INTRINSIC(EndScope, nullptr)                     ///< Dummy function which marks the end of a @p Scope.