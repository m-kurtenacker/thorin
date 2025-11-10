#include "lower_offload_intrinsics.h"

#include "runtime.h"

#include "thorin/offload/kernel_config.h"

namespace thorin {

struct RuntimeAPI {
    World& world_;
    Offload& backends_;

    const Def* anydsl_alloc;
    const Def* anydsl_alloc_unified;
    const Def* anydsl_release;
    const Def* anydsl_launch_kernel;
    const Def* anydsl_parallel_for;
    const Def* anydsl_fibers_spawn;
    const Def* anydsl_spawn_thread;
    const Def* anydsl_sync_thread;
    const Def* anydsl_create_graph;
    const Def* anydsl_create_task;
    const Def* anydsl_create_edge;
    const Def* anydsl_execute_graph;

    RuntimeAPI(World& world, Offload& backends) : world_(world), backends_(backends) {
        auto get_api_fn = [&](Types dom, const Type* codom, std::string name) {
            auto mem_ty = world.mem_type();
            auto found = world.find_cont(name.c_str());
            if (found)
                return found;
            auto r = codom ? world.return_type({mem_ty, codom}) : world.return_type({mem_ty});
            Array<const Type*> p = concat<const Type*>(mem_ty, concat<const Type*>(dom, r));
            auto c = world.continuation(world.fn_type(p), name);
            c->attributes_.cc = CC::C;
            world.make_external(c);
            return c;
        };

        auto i32 = world.type_qs32();
        auto i64 = world.type_qs64();
        auto ptr_ty = world.ptr_type(world.indefinite_array_type(world.type_pu8()));

        anydsl_alloc = get_api_fn({ i32, i64 }, ptr_ty, "anydsl_alloc");
        anydsl_alloc_unified = get_api_fn({ i32, i64 }, ptr_ty, "anydsl_alloc_unified");
        anydsl_release = get_api_fn({ i32 }, nullptr, "anydsl_release");
        anydsl_launch_kernel = get_api_fn({ i32, ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, i32 }, nullptr, "anydsl_launch_kernel");
        anydsl_parallel_for = get_api_fn({ i32, i32, i32, ptr_ty, ptr_ty }, nullptr, "anydsl_parallel_for");
        anydsl_fibers_spawn = get_api_fn({ i32, i32, i32, ptr_ty, ptr_ty }, nullptr, "anydsl_fibers_spawn");
        anydsl_spawn_thread = get_api_fn({ ptr_ty, ptr_ty }, i32, "anydsl_spawn_thread");
        anydsl_sync_thread = get_api_fn({ i32 }, nullptr, "anydsl_sync_thread");
        anydsl_create_graph = get_api_fn({ i32 }, i32, "anydsl_create_graph");
        anydsl_create_task = get_api_fn({ i32, world.tuple_type({ ptr_ty, i64 }) }, i32, "anydsl_create_task");
        anydsl_create_edge = get_api_fn({ i32, i32}, nullptr, "anydsl_create_edge");
        anydsl_execute_graph = get_api_fn({ i32, i32}, nullptr, "anydsl_execute_graph");
    }
};

static bool contains_ptrtype(const Type* type) {
    switch (type->tag()) {
        case Node_PtrType:             return false;
        case Node_IndefiniteArrayType: return contains_ptrtype(type->as<ArrayType>()->elem_type());
        case Node_DefiniteArrayType:   return contains_ptrtype(type->as<DefiniteArrayType>()->elem_type());
        case Node_FnType:              return false;
        case Node_StructType: {
            bool good = true;
            auto struct_type = type->as<StructType>();
            for (auto& t : struct_type->types())
                good &= contains_ptrtype(t);
            return good;
        }
        case Node_TupleType: {
            bool good = true;
            auto tuple = type->as<TupleType>();
            for (auto& t : tuple->types())
                good &= contains_ptrtype(t);
            return good;
        }
        default: return true;
    }
}

void emit_host_code(RuntimeAPI& api, const App* launch, Platform platform, Continuation* continuation, GetKernelConfigFn get_config) {
    World& world = continuation->world();

    assert(continuation->has_body());
    auto body = continuation->body();
    // to-target is the desired kernel call
    // target(mem, device, (dim.x, dim.y, dim.z), (block.x, block.y, block.z), body, return, free_vars)
    auto target = body->callee()->as_nom<Continuation>();
    assert_unused(target->is_intrinsic());
    assert(body->num_args() >= KernelLaunchArgs::Num && "required arguments are missing");

    // arguments
    const Def* mem = body->arg(KernelLaunchArgs::Mem);
    auto ret = body->arg(KernelLaunchArgs::Return);

    auto target_device_id = body->arg(KernelLaunchArgs::Device);
    auto target_platform = world.literal_qs32(static_cast<uint32_t>(platform), {});
    auto target_device = world.arithop_or(target_platform, world.arithop_shl(target_device_id, world.literal_qs32(4, {})));

    auto it_space = body->arg(KernelLaunchArgs::Space);
    auto it_config = body->arg(KernelLaunchArgs::Config);
    auto kernel = body->arg(KernelLaunchArgs::Body)->as_nom<Continuation>();

    //auto kernel_name = builder.CreateGlobalString(kernel->name() == "hls_top" ? kernel->name() : kernel->name());
    auto [fn, kn] = api.backends_.register_kernel_for_offloading(launch, kernel, get_config(launch, kernel));
    auto file_name = world.global_immutable_string(fn);
    auto kernel_name = world.global_immutable_string(kn);
    const size_t num_kernel_args = body->num_args() - KernelLaunchArgs::Num;

    auto ptr_ty = world.ptr_type(world.indefinite_array_type(world.type_pu8()));

    auto alloc = [&](const Type* t, std::string name) {
        auto a = world.alloc(t, mem, { name });
        mem = world.extract(a, static_cast<uint32_t>(0));
        return world.extract(a, 1);
    };

    auto store = [&](const Def* val, const Def* ptr) {
        mem = world.store(mem, ptr, val);
    };

    // allocate argument pointers, sizes, and types
    const Def* args   = alloc(world.definite_array_type(ptr_ty,                 num_kernel_args), "args");
    const Def* sizes  = alloc(world.definite_array_type(world.type_pu32(), num_kernel_args), "sizes");
    const Def* aligns = alloc(world.definite_array_type(world.type_pu32(), num_kernel_args), "aligns");
    const Def* allocs = alloc(world.definite_array_type(world.type_pu32(), num_kernel_args), "allocs");
    const Def* types  = alloc(world.definite_array_type(world.type_pu8(),  num_kernel_args), "types");

    // fill array of arguments
    for (size_t i = 0; i < num_kernel_args; ++i) {
        auto target_arg = body->arg(i + KernelLaunchArgs::Num);
        //const auto target_val = code_gen.emit(target_arg);
        auto target_val = target_arg;

        KernelArgType arg_type;
        const Def* void_ptr;
        if (target_arg->type()->isa<DefiniteArrayType>() ||
            target_arg->type()->isa<StructType>() ||
            target_arg->type()->isa<TupleType>()) {
            // definite array | struct | tuple
            auto alloca = alloc(target_arg->type(), target_arg->name());
            store(target_val, alloca);

            // check if argument type contains pointers
            if (!contains_ptrtype(target_arg->type()))
                world.wdef(target_arg, "argument '{}' of aggregate type '{}' contains pointer (not supported in OpenCL 1.2)", target_arg, target_arg->type());

            void_ptr = world.bitcast(ptr_ty, alloca);
            arg_type = KernelArgType::Struct;
        } else if (target_arg->type()->isa<PtrType>()) {
            auto ptr = target_arg->type()->as<PtrType>();
            auto rtype = ptr->pointee();

            //if (!rtype->isa<ArrayType>())
            //    world.edef(target_arg, "currently only pointers to arrays supported as kernel argument; argument has different type: {}", ptr);

            auto alloca = alloc(ptr_ty, target_arg->name());
            auto target_ptr = world.bitcast(ptr_ty, target_val);
            store(target_ptr, alloca);
            void_ptr = world.bitcast(ptr_ty, alloca);
            arg_type = KernelArgType::Ptr;
        } else {
            // normal variable
            auto alloca = alloc(target_arg->type(), target_arg->name());
            store(target_val, alloca);

            void_ptr = world.bitcast(ptr_ty, alloca);
            arg_type = KernelArgType::Val;
        }

        auto arg_ptr   = world.lea(args,   world.literal_pu32(i, {}), {});
        auto size_ptr  = world.lea(sizes,  world.literal_pu32(i, {}), {});
        auto align_ptr = world.lea(aligns, world.literal_pu32(i, {}), {});
        auto alloc_ptr = world.lea(allocs, world.literal_pu32(i, {}), {});
        auto type_ptr  = world.lea(types,  world.literal_pu32(i, {}), {});

        auto size = world.size_of(target_arg->type());

        //if (auto struct_type = llvm::dyn_cast<llvm::StructType>(target_val->getType())) {
        //    // In the case of a structure, do not include the padding at the end in the size
        //    auto last_elem   = struct_type->getStructNumElements() - 1;
        //    auto last_offset = layout_.getStructLayout(struct_type)->getElementOffset(last_elem);
        //    size = last_offset + layout_.getTypeStoreSize(struct_type->getStructElementType(last_elem)).getFixedValue();
        //}

        store(void_ptr, arg_ptr);
        store(size, size_ptr);
        store(world.align_of(target_arg->type()), align_ptr);
        store(world.size_of(target_arg->type()), alloc_ptr);
        store(world.literal_pu8((uint8_t)arg_type, {}), type_ptr);
    }

    // allocate arrays for the grid and block size
    const Def* grid_array = world.definite_array(world.type_qs32(), {
        world.extract(it_space, 0_u32),
        world.extract(it_space, 1_u32),
        world.extract(it_space, 2_u32),
    });
    const Def* grid_size = alloc(world.definite_array_type(world.type_qs32(), 3), "grid_size_alloc");
    store(grid_array, grid_size);

    const Def* block_array =  world.definite_array(world.type_qs32(), {
        world.extract(it_config, 0_u32),
        world.extract(it_config, 1_u32),
        world.extract(it_config, 2_u32),
    });
    const Def* block_size = alloc(world.definite_array_type(world.type_qs32(), 3), "block_size_alloc");
    store(block_array, block_size);

    grid_size  = world.bitcast(ptr_ty, grid_size);
    block_size = world.bitcast(ptr_ty, block_size);
    args       = world.bitcast(ptr_ty, args);
    sizes      = world.bitcast(ptr_ty, sizes);
    aligns     = world.bitcast(ptr_ty, aligns);
    allocs     = world.bitcast(ptr_ty, allocs);
    types      = world.bitcast(ptr_ty, types);

    file_name = world.bitcast(ptr_ty, file_name);
    kernel_name = world.bitcast(ptr_ty, kernel_name);

    continuation->set_body(world.app(api.anydsl_launch_kernel, {mem, target_device, file_name, kernel_name, grid_size, block_size, args, sizes, aligns, allocs, types, world.literal_qs32(num_kernel_args, {}), ret}));
}

std::tuple<const Def*, Array<const Def*>> spill(const Def*& mem, const Defs& args, const Def*& wrapper_mem, const Def* wrapper_ptr) {
    World& world = mem->world();
    StructType* st = world.struct_type("spillbox", args.size());
    for (size_t i = 0; i < args.size(); i++)
        st->set_op(i, args[i]->type());

    auto alloc = [&](const Type* t, std::string name) {
        auto a = world.alloc(t, mem, { name });
        mem = world.extract(a, static_cast<uint32_t>(0));
        return world.extract(a, 1);
    };

    auto store = [&](const Def* val, const Def* ptr) {
        mem = world.store(mem, ptr, val);
    };

    const Def* spill_alloca = alloc(st, "spill");
    std::vector<const Def*> restored;
    wrapper_ptr = world.bitcast(spill_alloca->type(), wrapper_ptr);
    for (size_t i = 0; i < args.size(); i++) {
        store(args[i], world.lea(spill_alloca, world.literal_pu32(i, {}), {}));
        auto l = world.load(wrapper_mem, world.lea(wrapper_ptr, world.literal_pu32(i, {}), {}));
        wrapper_mem = world.extract(l, static_cast<uint32_t>(0));
        restored.push_back(world.extract(l, 1));
    }

    auto ptr_ty = world.ptr_type(world.indefinite_array_type(world.type_pu8()));
    return std::make_tuple(world.bitcast(ptr_ty, spill_alloca), restored);
}

enum class RuntimeParallelForArgs {
    Mem = 0,
    NumThreads,
    Lower,
    Upper,
    Args,
    Fun,
    Return,
};

void emit_parallel(RuntimeAPI& api, Continuation* continuation) {
    World& world = continuation->world();
    auto ptr_ty = world.ptr_type(world.indefinite_array_type(world.type_pu8()));

    assert(continuation->has_body());
    auto body = continuation->body();
    const Def* mem = body->arg(static_cast<size_t>(ParallelForArgs::Mem));
    auto numthreads = body->arg(static_cast<size_t>(ParallelForArgs::NumThreads));
    auto lower = body->arg(static_cast<size_t>(ParallelForArgs::Lower));
    auto upper = body->arg(static_cast<size_t>(ParallelForArgs::Upper));
    auto fun = body->arg(static_cast<size_t>(ParallelForArgs::Fun))->as<Continuation>();
    auto ret = body->arg(static_cast<size_t>(ParallelForArgs::Return));

    // create loop iterating over range:
    // for (int i=lower; i<upper; ++i)
    //   body(i, <closure_elems>);

    auto wrapper = world.continuation(world.fn_type({world.mem_type(), ptr_ty, world.type_qs32(), world.type_qs32(), world.return_type({world.mem_type()})}));
    world.make_external(wrapper);
    const Def* wrapper_mem = wrapper->mem_param();
    auto inner_lower = wrapper->param(2);
    auto inner_upper = wrapper->param(3);
    auto [args, recovered] = spill(mem, body->args().skip_front(static_cast<size_t>(ParallelForArgs::Num)), wrapper_mem, wrapper->param(1));

    auto loop_head = world.continuation(world.fn_type({world.mem_type(), world.type_qs32()}), "loop_head");
    auto loop_body = world.continuation(world.fn_type({world.mem_type()}), "loop_body");
    auto loop_continue = world.continuation(world.fn_type({world.mem_type()}), "loop_continue");
    auto loop_exit = world.continuation(world.fn_type({world.mem_type()}), "loop_exit");
    loop_head->branch(loop_head->mem_param(), world.cmp_lt(loop_head->param(1), inner_upper), loop_body, loop_exit);

    Array<const Def*> prefix {loop_body->mem_param(), world.bitcast(fun->param(1)->type(), loop_head->param(1)), world.return_point(loop_continue) };
    loop_body->jump(fun, concat(prefix, recovered));
    loop_continue->jump(loop_head, { loop_continue->mem_param(), world.arithop_add(loop_head->param(1), world.literal_qs32(1, {})) });
    loop_exit->jump(wrapper->ret_param(), loop_exit->params_as_defs());
    wrapper->jump(loop_head, { wrapper->mem_param(), inner_lower });

    continuation->set_body(world.app(api.anydsl_parallel_for, {mem, numthreads, lower, upper, world.bitcast(ptr_ty, args), world.bitcast(ptr_ty, wrapper), ret}));
}

enum class RuntimeSpawnFibersArgs {
    Mem = 0,
    NumThreads,
    NumBlocks,
    NumWarps,
    Args,
    Fun,
    Return,
};

void emit_fibers(RuntimeAPI& api, Continuation* continuation) {
    World& world = continuation->world();
    auto ptr_ty = world.ptr_type(world.type_pu8());
    auto i32 = world.type_qs32();

    assert(continuation->has_body());
    auto body = continuation->body();
    const Def* mem = body->arg(static_cast<size_t>(SpawnFibersArgs::Mem));
    auto threads = body->arg(static_cast<size_t>(SpawnFibersArgs::NumThreads));
    auto blocks = body->arg(static_cast<size_t>(SpawnFibersArgs::NumBlocks));
    auto warps = body->arg(static_cast<size_t>(SpawnFibersArgs::NumWarps));
    auto fun = body->arg(static_cast<size_t>(SpawnFibersArgs::Fun))->as<Continuation>();
    auto ret = body->arg(static_cast<size_t>(SpawnFibersArgs::Return));

    auto wrapper = world.continuation(world.fn_type({world.mem_type(), ptr_ty, i32, i32, world.return_type({world.mem_type()})}));
    const Def* wrapper_mem = wrapper->mem_param();
    auto [args, recovered] = spill(mem, body->args().skip_front(static_cast<size_t>(SpawnThreadArgs::Num)), wrapper_mem, wrapper->param(1));
    Array<const Def*> prefix = {mem, wrapper->param(2), wrapper->param(3)};
    wrapper->jump(fun, concat<const Def*>(concat(prefix, recovered), {wrapper->ret_param()}));

    continuation->set_body(world.app(api.anydsl_fibers_spawn, {mem, threads, blocks, warps, args, fun, ret}));
}

enum class RuntimeSpawnThreadArgs {
    Mem = 0,
    Args,
    Fun,
    Return,
};

void emit_spawn(RuntimeAPI& api, Continuation* continuation) {
    World& world = continuation->world();
    auto ptr_ty = world.ptr_type(world.indefinite_array_type(world.type_pu8()));

    assert(continuation->has_body());
    auto body = continuation->body();
    const Def* mem = body->arg(static_cast<size_t>(SpawnThreadArgs::Mem));
    auto fun = body->arg(static_cast<size_t>(SpawnThreadArgs::Fun));
    auto ret = body->arg(static_cast<size_t>(SpawnThreadArgs::Return));

    auto wrapper = world.continuation(world.fn_type({world.mem_type(), ptr_ty, world.return_type({world.mem_type()})}));
    const Def* wrapper_mem = wrapper->mem_param();
    auto [args, recovered] = spill(mem, body->args().skip_front(static_cast<size_t>(SpawnThreadArgs::Num)), wrapper_mem, wrapper->param(1));
    wrapper->jump(fun, concat<const Def*>(concat<const Def*>(mem, recovered.ref()), {wrapper->ret_param()}));

    continuation->set_body(world.app(api.anydsl_sync_thread, {mem, world.bitcast(ptr_ty, args), world.bitcast(ptr_ty, wrapper), ret}));
}

void emit_sync(RuntimeAPI& api, Continuation* continuation) {
    World& world = continuation->world();

    assert(continuation->has_body());
    auto body = continuation->body();
    const Def* mem = body->arg(static_cast<size_t>(SyncArgs::Mem));
    auto id = body->arg(static_cast<size_t>(SyncArgs::Id));
    auto ret = body->arg(static_cast<size_t>(SyncArgs::Return));

    continuation->set_body(world.app(api.anydsl_sync_thread, {mem, id, ret}));
}

auto build_setup_args_fn(World& world, const Def*& mem, ArrayRef<const Def*> args) {
    auto ptr_ty = world.ptr_type(world.indefinite_array_type(world.type_pu8()));
    auto callback_fn_t = world.closure_type({world.mem_type(),
        world.type_pu64(), world.type_qs32(),
        world.ptr_type(world.indefinite_array_type(ptr_ty)),
        world.ptr_type(world.indefinite_array_type(world.type_pu64())),
        world.return_type({world.mem_type()})});
    Continuation* setup_args_fn = world.continuation(
            world.fn_type({world.mem_type(), world.type_pu64(), callback_fn_t, world.return_type({world.mem_type()})}));
    //const Def* mem = setup_args_fn->mem_param();
    auto pipeline_handle = setup_args_fn->param(1);
    auto callback = setup_args_fn->param(2);
    auto alloc = [&](const Type* t) {
        auto r = world.alloc(t, mem);
        mem = world.extract(r, static_cast<uint32_t>(0));
        return world.extract(r, static_cast<uint32_t>(1));
    };
    auto pointers = alloc(world.definite_array_type(ptr_ty, args.size()));
    std::vector<const Def*> sizes;
    for (size_t i = 0; i < args.size(); i++) {
        auto arg_on_stack = alloc(args[i]->type());
        mem = world.store(mem, arg_on_stack, args[i]);
        mem = world.store(mem, world.lea(pointers, world.literal_pu32(i, {}), {}), world.bitcast(ptr_ty, arg_on_stack));
        sizes.push_back(world.size_of(args[i]->type()));
    }
    auto sizes_global = world.global(world.definite_array(world.type_qs64(), sizes), false, {"sizes"});
    auto dummy_closure = world.closure(world.closure_type(setup_args_fn->type()->types()));
    auto self_param = setup_args_fn->append_param(dummy_closure->type());
    auto env = world.closure_env(pointers->type(), setup_args_fn->mem_param(), self_param);
    auto restored_pointers = world.bitcast(callback_fn_t->types()[3], world.extract(env, 1));
    auto sizes_cast = world.bitcast(callback_fn_t->types()[4], sizes_global);
    setup_args_fn->set_body(world.app(callback, { world.extract(env, 0u), pipeline_handle, world.literal_qs32(args.size(), {}), restored_pointers, sizes_cast, setup_args_fn->ret_param() }));
    // setup function has to be a closure
    dummy_closure->set_fn(setup_args_fn, self_param->index());
    dummy_closure->set_env(pointers);
    return dummy_closure;
};

#ifdef THORIN_ENABLE_SPIRV
void emit_vulkan_offload(RuntimeAPI& api, Continuation* continuation) {
    World& world = continuation->world();
    auto ptr_ty = world.ptr_type(world.indefinite_array_type(world.type_pu8()));

    assert(continuation->has_body());
    auto body = continuation->body();
    const Def* mem = body->arg(0);
    auto ret = body->ret_arg();

    auto num_stages = body->arg(1)->as<PrimLit>()->value().get_u32();

    std::vector<const Def*> stages;
    for (size_t stage = 0; stage < num_stages; stage++) {
        auto shader_type = body->arg(2 + stage * 2 + 0);
        auto shader_code = body->arg(2 + stage * 2 + 1)->as_nom<Continuation>();
        ShaderKernelConfig kc;
        kc.execution_model_ = static_cast<SpvExecutionModel>(primlit_value<uint32_t>(shader_type));
        auto [fn, kn] = api.backends_.register_kernel_for_offloading(body, shader_code, std::make_unique<ShaderKernelConfig>(kc));
        stages.push_back(world.tuple({shader_type, world.bitcast(ptr_ty, world.global_immutable_string(fn)), world.bitcast(ptr_ty, world.global_immutable_string(kn)) }));
    }
    auto stages_definite = world.definite_array(world.tuple_type({ world.type_pu32(), ptr_ty, ptr_ty }), stages);
    auto stages_global = world.global(stages_definite, true);

    auto struct_t = ret->type()->as<ReturnType>()->types()[1]->as<StructType>();
    std::vector<const Def*> agg;

    agg.push_back(build_setup_args_fn(world, mem, body->args().skip_front(3 + 2 * num_stages)));
    agg.push_back(world.literal_pu32(num_stages, {}));
    agg.push_back(world.bitcast(world.ptr_type(world.indefinite_array_type(world.tuple_type({ world.type_pu32(), ptr_ty, ptr_ty }))), stages_global));

    continuation->set_body(world.app(ret, {mem, world.struct_agg(struct_t, agg)}));
    //continuation->set_body(world.app(api.anydsl_sync_thread, {mem, id, ret}));
}
#endif

void lower_offload_intrinsics(World& world, Offload& backends) {
    RuntimeAPI api(world, backends);

    for (auto continuation : world.copy_continuations()) {
        if (!continuation->has_body()) continue;
        auto call = continuation->body();
        if (auto callee = call->callee()->isa<Continuation>()) {
            switch (callee->intrinsic()) {
                case Intrinsic::CUDA:                emit_host_code(api, call, Platform::CUDA_PLATFORM,       continuation, get_compute_kernel_config); break;
                case Intrinsic::NVVM:                emit_host_code(api, call, Platform::CUDA_PLATFORM,       continuation, get_compute_kernel_config); break;
                case Intrinsic::OpenCL:              emit_host_code(api, call, Platform::OPENCL_PLATFORM,     continuation, get_compute_kernel_config); break;
                case Intrinsic::OpenCL_SPIRV:        emit_host_code(api, call, Platform::OPENCL_PLATFORM,     continuation, get_compute_kernel_config); break;
                case Intrinsic::LevelZero_SPIRV:     emit_host_code(api, call, Platform::LEVEL_ZERO_PLATFORM, continuation, get_compute_kernel_config); break;
                case Intrinsic::AMDGPUHSA:           emit_host_code(api, call, Platform::HSA_PLATFORM,        continuation, get_compute_kernel_config); break;
                case Intrinsic::AMDGPUPAL:           emit_host_code(api, call, Platform::PAL_PLATFORM,        continuation, get_compute_kernel_config); break;
                case Intrinsic::VulkanCS_SPIRV:      emit_host_code(api, call, Platform::VULKAN_PLATFORM,     continuation, get_compute_kernel_config); break;
                case Intrinsic::VulkanOffload_SPIRV: emit_vulkan_offload(api, continuation); break;

                case Intrinsic::Parallel:        emit_parallel(api, continuation); break;
                case Intrinsic::Fibers:          emit_fibers(api, continuation);   break;
                case Intrinsic::Spawn:           emit_spawn(api, continuation);    break;
                case Intrinsic::Sync:            emit_sync(api, continuation);     break;
                default: continue;
            }
        }
    }
}

}
