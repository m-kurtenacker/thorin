#include "compare.h"

#include "thorin/transform/importer.h"

namespace thorin {

struct Comparator {
    World& world_;
    Def2Def b_to_a_;
    Def2Def a_to_b_;

    DefSet cache_;

    Comparator(World& world) : world_(world) {}

    bool fail(const Def* a, const Def* b, std::string message) {
        world_.edef(a, " is different from {}, {}", b, message);
        return false;
    }

    bool compare_defs(const Def* a, const Def* b) {
        if (a == b)
            return true;
        // both nodes must be null or neither can be
        if ((a != nullptr) != (b != nullptr))
            return fail(a, b, "one pointer is null but the other isn't");
        assert(&a->world() == &world_);
        assert(&a->world() == &b->world());
        if (cache_.contains(a))
            return true;
        if (a->tag() != b->tag())
            return fail(a, b, "tags do not match");
        if (a->num_ops() != b->num_ops())
            return fail(a, b, "number of ops does not match");
        if (!compare_defs(a->type(), b->type()))
            return fail(a, b, "type does not match");

        // assume two nominal nodes are the same, but check for consistency
        if (a->isa_nom()) {
            // if we've seen either before and it corresponds to something else, fail the comparison
            auto a_prime = b_to_a_.find(b);
            if (a_prime != b_to_a_.end() && a_prime->second != a)
                return fail(a, b, "nominal mismatch");
            auto b_prime = a_to_b_.find(a);
            if (b_prime != a_to_b_.end() && b_prime->second != b)
                return fail(a, b, "nominal mismatch");
            // at this point we either have a match or not!
            assert ((a_prime != b_to_a_.end()) == (b_prime != a_to_b_.end()));
            // if it's a match, we now assume they're identical
            if (a_prime != b_to_a_.end()) {
                cache_.insert(a);
                return true;
            } else {
                // if not, introduce the constraint
                b_to_a_[b] = a;
                a_to_b_[a] = b;
            }
        }

        for (size_t i = 0; i < a->num_ops(); i++) {
            if (!compare_defs(a->op(i), b->op(i)))
                return fail(a, b, "op "+std::to_string(i)+" does not match");
        }

        // TODO: add more...
        if (auto param_a = a->isa<Param>()) {
            if (param_a->index() != b->as<Param>()->index())
                return fail(a, b, "param index does not match");
        } else if (auto variant_a = a->isa<Variant>()) {
            if (variant_a->index() != b->as<Variant>()->index())
                return fail(a, b, "variant index does not match");
        } else if (auto def_arr_t_a = a->isa<DefiniteArrayType>()) {
            if (def_arr_t_a->dim() != b->as<DefiniteArrayType>()->dim())
                return fail(a, b, "array size does not match");
        } else if (auto ptr_type_a = a->isa<PtrType>()) {
            if (ptr_type_a->addr_space() != b->as<PtrType>()->addr_space())
                return fail(a, b, "address space does not match");
        }

        cache_.insert(a);
        return true;
    }
};

bool compare_defs(const Def* a, const Def* b) {
    Comparator cmp(a->world());
    return cmp.compare_defs(a, b);
}

bool compare_worlds(World& world_a, World& world_b) {
    std::unique_ptr<World> world = std::make_unique<World>(world_a);
    Comparator cmp(*world);

    std::unordered_map<std::string, std::tuple<const Def*, const Def*>> externals;
    Importer a_importer(world_a, *world);
    for (auto& [name, external] : world_a.externals()) {
        auto& [a, b] = externals[external->name()];
        a = a_importer.import(external);
    }
    Importer b_importer(world_b, *world);
    for (auto& [name, external] : world_b.externals()) {
        auto& [a, b] = externals[external->name()];
        b = b_importer.import(external);
    }

    for (auto& [name, a_b] : externals) {
        auto [a, b] = a_b;
        assert(a || b && "two nulls is not expected");
        if (!a)
            return cmp.fail(b, a, "external def does not exist in A");
        if (!b)
            return cmp.fail(a, b, "external def does not exist in B");
        if (!cmp.compare_defs(a, b))
            return false;
    }
    return true;
}

}
