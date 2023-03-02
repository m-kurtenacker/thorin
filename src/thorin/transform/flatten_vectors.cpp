#include "thorin/transform/flatten_vectors.h"
#include "thorin/continuation.h"
#include "thorin/world.h"
#include "thorin/transform/mangle.h"
#include "thorin/analyses/verify.h"
#include "thorin/analyses/schedule.h"

#include <limits>

namespace thorin {

class Flatten {
    Def2Def def2def;
    Type2Type type2type;
    World& world;

    bool contains_vector_container(const Type*);

    const Type* flatten_type(const Type*);
    const Type* flatten_vector_type(const VectorContainerType*);
    const FnType* flatten_fn_type(const FnType *);

    const Def * flatten_def(const Def *);
    const Def * flatten_primop(const Def *);
    Continuation* flatten_continuation(const Continuation*);
    void flatten_body(const Continuation *, Continuation *);

public:
    Flatten(World &world) : world(world) {};
    void run();
};

bool Flatten::contains_vector_container(const Type* type) {
    if (type->isa<VectorContainerType>())
        return true;
    else {
        for (auto op : type->ops()) {
            if (contains_vector_container(op->as<Type>()))
                return true;
        }
    }
    return false;
}

const Type* Flatten::flatten_vector_type(const VectorContainerType *vector_type) {
    auto vector_length = vector_type->length();
    auto element_type = vector_type->element();

    if (auto nominal_type = element_type->isa<NominalType>()) {
        assert(nominal_type->isa<StructType>()); //Only this is supported RN.
        
        StructType* new_struct_type = world.struct_type(nominal_type->name() + "_flattened", nominal_type->num_ops());
        type2type[vector_type] = new_struct_type;
        for (size_t i = 0; i < nominal_type->num_ops(); ++i) {
            auto inner_vector_type = world.vec_type(nominal_type->op(i)->as<Type>(), vector_length);
            new_struct_type->set_op(i, flatten_type(inner_vector_type));
            new_struct_type->set_op_name(i, nominal_type->op_name(i));
        }
        return new_struct_type;
    } else if (auto tuple_type = element_type->isa<TupleType>(); tuple_type && tuple_type->num_ops() == 0) {
        return tuple_type;
    } else {
        element_type->dump();
        THORIN_UNREACHABLE;
    }
}

const Type* Flatten::flatten_type(const Type *type) {
    if (!contains_vector_container(type))
        return type;

    auto flattened = type2type[type];
    if (flattened)
        return flattened;

    if (auto vector = type->isa<VectorContainerType>()) {
        return type2type[type] = flatten_vector_type(vector);
    } else if (auto nominal_type = type->isa<NominalType>()) {
        assert(nominal_type->isa<StructType>()); //Only this is supported RN.

        StructType* new_struct_type = world.struct_type(nominal_type->name() + "_flat", nominal_type->num_ops());
        type2type[type] = new_struct_type;
        for (size_t i = 0; i < nominal_type->num_ops(); ++i) {
            new_struct_type->set_op(i, flatten_type(nominal_type->op(i)->as<Type>()));
            new_struct_type->set_op_name(i, nominal_type->op_name(i));
        }
        return new_struct_type;
    } else {
        Array<const Def*> new_ops(type->num_ops());
        for (size_t i = 0; i < type->num_ops(); i++) {
            auto op = type->op(i);
            auto new_op = flatten_type(op->as<Type>());
            new_ops[i] = new_op;
        }
        auto new_type = type->rebuild(world, nullptr, new_ops);
        return type2type[type] = new_type->as<Type>();
    }
}


const FnType* Flatten::flatten_fn_type(const FnType *fntype) {
    assert(false);
    std::vector<const Type*> arg_types;
    for (auto op : fntype->ops()) {
        const Type* result = nullptr;
        if (auto vecextended = op->isa<VectorContainerType>())
            result = flatten_type(vecextended);
        else if (auto fn = op->isa<FnType>())
            result = flatten_fn_type(fn);
        else
            result = op->as<Type>();
        arg_types.emplace_back(result);
    }
    return world.fn_type(arg_types);
}

const Def * Flatten::flatten_def(const Def *def) {
    auto replacement = def2def[def];
    if (replacement)
        return replacement;

    if (auto cont = def->isa<Continuation>(); cont && (cont->is_intrinsic() || cont->empty())) {
        return cont;
        if(cont == world.branch())
            return cont;
        Debug de = cont->debug();
        if (de.name == "predicated") {
            auto new_type = flatten_fn_type(cont->type());
            return world.continuation(new_type, cont->attributes(), de);
        } else {
            auto new_type = flatten_type(cont->type());
            return world.continuation(new_type->as<FnType>(), cont->attributes(), de);
        }
    } else if (auto cont = def->isa<Continuation>(); cont && !cont->is_intrinsic()) {
        auto new_continuation = flatten_continuation(cont);
        return new_continuation;
    } else if (auto param = def->isa<Param>()) {
        flatten_continuation(param->continuation());
        auto replacement = def2def[def];
        if (replacement)
            return replacement;
        assert(false && "Parameters should be handled beforehand!");
    } else {
        return flatten_primop(def);
    }
}

const Def * Flatten::flatten_primop(const Def *primop) {
    auto replacement = def2def[primop];
    if (replacement)
        return replacement;

    auto primop_type = primop->type();
    assert(primop_type);
    const Type* newtype;
    newtype = flatten_type(primop_type);
    assert(newtype);

    Array<const Def*> nops(primop->num_ops());

    for (size_t i = 0, e = primop->num_ops(); i != e; ++i) {
        nops[i] = flatten_def(primop->op(i));
    }

    const Def* new_primop;

    if (primop->isa<PrimLit>()) {
        new_primop = primop;
    } else if (auto store = primop->isa<Store>()) {
        if (store->val()->type() == nops[2]->type() &&
            store->ptr()->type() == nops[1]->type())
            new_primop = primop->rebuild(world, newtype, nops);
        else {
            auto mem = nops[0];
            Array<const Def*> elements(nops[1]->type()->as<PtrType>()->pointee()->num_ops());
            for (size_t i = 0; i < nops[1]->type()->as<PtrType>()->pointee()->num_ops(); ++i) {
                auto element_ptr = world.lea(nops[1], world.literal_qu32(i, {}), {});
                auto val = world.extract(nops[2], i);
                mem = world.store(mem, element_ptr, val);
            }
            new_primop = mem;
        }
    } else if (auto load = primop->isa<Load>()) {
        if (load->type() != newtype) {
            auto mem = nops[0];
            Array<const Def*> elements(nops[1]->type()->as<PtrType>()->pointee()->num_ops());
            for (size_t i = 0; i < nops[1]->type()->as<PtrType>()->pointee()->num_ops(); ++i) {
                auto element_ptr = world.lea(nops[1], world.literal_qu32(i, {}), {});
                auto load = world.load(mem, element_ptr);
                mem = world.extract(load, (u32) 0);
                auto element = world.extract(load, 1);
                elements[i] = element;
            }
            auto aggregate = world.struct_agg(newtype->op(1)->as<StructType>(), elements);
            new_primop = world.tuple({mem, aggregate});
        } else
            new_primop = primop->rebuild(world, newtype, nops);
    } else if (auto extract = primop->isa<Extract>()) {
        if (nops[1]->type()->isa<TupleType>()) {
            auto elem = world.extract(nops[1], 1);
            new_primop = primop->rebuild(world, newtype, { nops[0], elem });
        } else
            new_primop = primop->rebuild(world, newtype, nops);
    } else if (primop->isa<VectorLift>()) {
        new_primop = nops[0];
    } else {
        new_primop = primop->rebuild(world, newtype, nops);
    }

    assert(new_primop);

    def2def[primop] = new_primop;

    return new_primop;
}

void Flatten::flatten_body(const Continuation *old_continuation, Continuation *new_continuation) {
    auto old_app = old_continuation->body();

    const Def* ntarget = flatten_def(old_app->callee());

    Array<const Def*>nargs(ntarget->type()->num_ops());
    for (size_t i = 0, e = old_app->num_args(); i != e; ++i)
        nargs[i] = flatten_def(old_app->arg(i));

    new_continuation->jump(ntarget, nargs, old_continuation->debug());
}

Continuation* Flatten::flatten_continuation(const Continuation* kernel) {
    auto replacement = def2def[kernel];
    if (replacement)
        return const_cast<Continuation*>(replacement->as<Continuation>());;

    auto new_type = flatten_type(kernel->type());

    Continuation *ncontinuation;
    ncontinuation = world.continuation(new_type->as<FnType>(), kernel->debug_history());

    def2def[kernel] = ncontinuation;

    for (size_t i = 0, j = 0, e = kernel->num_params(); i != e; ++i) {
        auto old_param = kernel->param(i);
        auto new_param = ncontinuation->param(j++);
        assert(old_param);
        assert(new_param);
        def2def[old_param] = new_param;
        new_param->set_name(old_param->name());
    }

#if 0
    // mangle filter
    if (!kernel->filter()->is_empty()) {
        Array<const Def*> new_conditions(ncontinuation->num_params());
        size_t j = 0;
        for (size_t i = 0, e = kernel->num_params(); i != e; ++i) {
            new_conditions[j++] = flatten_def(kernel->filter()->condition(i));
        }

        for (size_t e = ncontinuation->num_params(); j != e; ++j)
            new_conditions[j] = world.literal_bool(false, Debug{});

        ncontinuation->set_filter(world.filter(new_conditions, kernel->filter()->debug()));
    }
#endif

    return ncontinuation;
}

void Flatten::run() {
    for (auto continuation : world.copy_continuations()) {
        if (!continuation->has_body())
            continue;
        std::cerr << "Analyzing " << continuation->unique_name();
        std::cerr << "\n";

        Continuation* new_continuation = flatten_continuation(continuation);
        flatten_body(continuation, new_continuation);
        if (continuation->is_exported()) {
            new_continuation->name() == continuation->name();
            world.make_external(new_continuation);
        }

        //Continuation *newb = const_cast<Continuation*>(new_continuation);
        //continuation->replace_uses(newb);
    }
}

void flatten_vectors(Thorin& thorin) {
    thorin.world().VLOG("start flatten");

    Flatten(thorin.world()).run();

    thorin.world().VLOG("end flatten");
    std::cerr << "end flatten\n";
}

}
