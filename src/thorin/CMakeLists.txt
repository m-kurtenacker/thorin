set(THORIN_SOURCES
    ../../include/thorin/analyses/cfg.h
    ../../include/thorin/analyses/domtree.h
    ../../include/thorin/analyses/free_defs.h
    ../../include/thorin/analyses/looptree.h
    ../../include/thorin/analyses/schedule.h
    ../../include/thorin/analyses/scope.h
    ../../include/thorin/analyses/verify.h
    ../../include/thorin/be/c/c.h
    ../../include/thorin/be/codegen.h
    ../../include/thorin/be/emitter.h
    ../../include/thorin/be/json/json.h
    ../../include/thorin/be/kernel_config.h
    ../../include/thorin/be/llvm/amdgpu.h
    ../../include/thorin/be/llvm/amdgpu_hsa.h
    ../../include/thorin/be/llvm/amdgpu_pal.h
    ../../include/thorin/be/llvm/cpu.h
    ../../include/thorin/be/llvm/llvm.h
    ../../include/thorin/be/llvm/nvvm.h
    ../../include/thorin/be/llvm/runtime.h
    ../../include/thorin/tables/allnodes.h
    ../../include/thorin/tables/arithoptable.h
    ../../include/thorin/tables/cmptable.h
    ../../include/thorin/tables/mathoptable.h
    ../../include/thorin/tables/nodetable.h
    ../../include/thorin/tables/primtypetable.h
    ../../include/thorin/transform/closure_conversion.h
    ../../include/thorin/transform/codegen_prepare.h
    ../../include/thorin/transform/dead_load_opt.h
    ../../include/thorin/transform/flatten_tuples.h
    ../../include/thorin/transform/hls_channels.h
    ../../include/thorin/transform/hls_kernel_launch.h
    ../../include/thorin/transform/hoist_enters.h
    ../../include/thorin/transform/importer.h
    ../../include/thorin/transform/inliner.h
    ../../include/thorin/transform/lift_builtins.h
    ../../include/thorin/transform/mangle.h
    ../../include/thorin/transform/partial_evaluation.h
    ../../include/thorin/transform/plugin_execute.h
    ../../include/thorin/transform/resolve_loads.h
    ../../include/thorin/transform/rewrite.h
    ../../include/thorin/transform/split_slots.h
    ../../include/thorin/util/array.h
    ../../include/thorin/util/cast.h
    ../../include/thorin/util/hash.h
    ../../include/thorin/util/indexmap.h
    ../../include/thorin/util/indexset.h
    ../../include/thorin/util/iterator.h
    ../../include/thorin/util/scoped_dump.h
    ../../include/thorin/util/stream.h
    ../../include/thorin/util/symbol.h
    ../../include/thorin/util/types.h
    ../../include/thorin/util/utility.h
    ../../include/thorin/config.h.in
    ../../include/thorin/continuation.h
    ../../include/thorin/debug.h
    ../../include/thorin/def.h
    ../../include/thorin/enums.h
    ../../include/thorin/primop.h
    ../../include/thorin/type.h
    ../../include/thorin/world.h

    continuation.cpp
    debug.cpp
    def.cpp
    enums.cpp
    primop.cpp
    rec_stream.cpp
    type.cpp
    world.cpp
    analyses/cfg.cpp
    analyses/domtree.cpp
    analyses/free_defs.cpp
    analyses/looptree.cpp
    analyses/schedule.cpp
    analyses/scope.cpp
    analyses/verify.cpp
    be/codegen.cpp
    be/c/c.cpp
    transform/cleanup_world.cpp
    transform/closure_conversion.cpp
    transform/codegen_prepare.cpp
    transform/dead_load_opt.cpp
    transform/hls_channels.cpp
    transform/hls_kernel_launch.cpp
    transform/hoist_enters.cpp
    transform/flatten_tuples.cpp
    transform/importer.cpp
    transform/inliner.cpp
    transform/lift_builtins.cpp
    transform/mangle.cpp
    transform/partial_evaluation.cpp
    transform/rewrite.cpp
    transform/plugin_execute.cpp
    transform/resolve_loads.cpp
    transform/split_slots.cpp
    util/hash.cpp
    util/stream.cpp
    util/symbol.cpp
    util/graphviz_dump.cpp
    util/scoped_dump.cpp
    )

if(LLVM_FOUND)
    list(APPEND THORIN_SOURCES
        be/llvm/cpu.cpp
        be/llvm/llvm.cpp
        be/llvm/amdgpu.cpp
        be/llvm/amdgpu_hsa.cpp
        be/llvm/amdgpu_pal.cpp
        be/llvm/nvvm.cpp
        be/llvm/parallel.cpp
        be/llvm/runtime.inc
        be/llvm/runtime.cpp
        be/llvm/vectorize.cpp
    )
endif()

if (THORIN_ENABLE_SHADY)
    list(APPEND THORIN_SOURCES
        be/shady/shady.cpp
    )
endif()

if(THORIN_ENABLE_JSON)
    list(APPEND THORIN_SOURCES
        be/json/json.cpp
    )
endif()

add_library(thorin ${THORIN_SOURCES})
target_include_directories(thorin PUBLIC ${Half_INCLUDE_DIRS} ${Thorin_ROOT_DIR}/src ${Thorin_ROOT_DIR}/include ${CMAKE_BINARY_DIR}/include)

if(LLVM_FOUND)
    set(Thorin_LLVM_COMPONENTS core support ipo target ${LLVM_TARGETS_TO_BUILD})
    target_include_directories(thorin SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(thorin PRIVATE ${LLVM_DEFINITIONS})
    if(RV_FOUND)
        target_include_directories(thorin PRIVATE ${RV_INCLUDE_DIRS})
        target_link_libraries(thorin PRIVATE ${RV_LIBRARIES})
        list(APPEND Thorin_LLVM_COMPONENTS analysis passes transformutils)
    endif()
    llvm_config(thorin ${AnyDSL_LLVM_LINK_SHARED} ${Thorin_LLVM_COMPONENTS})
endif()

if (THORIN_ENABLE_SHADY)
    if (shady_FOUND)
        target_link_libraries(thorin PRIVATE shady::shady)
    else()
        target_link_libraries(thorin PRIVATE shady)
    endif()
endif()

if(THORIN_ENABLE_JSON)
    target_link_libraries(thorin PRIVATE nlohmann_json::nlohmann_json)
endif()

target_link_libraries(thorin PRIVATE ${CMAKE_DL_LIBS})
