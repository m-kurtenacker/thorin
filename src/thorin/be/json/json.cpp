#include "json.h"

namespace thorin::json {

class DefTable {
public:
    json defs = json::object();

    DefMap<std::string> known_defs;

    std::unordered_map<std::string, int> actually_unique_names;
    std::string unique_name(std::string name) {
        while (true) {
            auto it = actually_unique_names.find(name);
            if (it == actually_unique_names.end()) {
                actually_unique_names[name] = 0;
                return name;
            }
            int suffix = ++it->second;
            name += "_";
            name += std::to_string(suffix);
        }
    }

    std::string def_name(const Def* def) {
        if (!def->name().empty())
            return unique_name(def->name());
        return unique_name(tag2str(def->tag()));
    }

    std::string translate_def_with_ops(const Def* def, std::vector<std::string>& ops, json& result) {
        // this uses the 'ops' argument as a cache of sorts
        auto translate_def = [&](const Def* op) -> std::string {
            for (size_t i = 0; i < def->num_ops(); i++) {
                if (def->op(i) == op)
                    return ops[i];
            }
            return DefTable::translate_def(op);
        };

        result["tag"] = tag2str(def->tag());
        if (!def->isa<Star>())
            result["def_type"] = translate_def(def->type());
        if (auto def_arr = def->isa<DefiniteArrayType>()) {
            auto elem_type = translate_def(def_arr->elem_type());
            result["args"] = { elem_type };
            result["length"] = def_arr->dim();
        } else if (auto indef_arr = def->isa<IndefiniteArrayType>()) {
            auto elem_type = translate_def(indef_arr->elem_type());
            result["args"] = { elem_type };
        } else if (def->isa<Star>()) {
        } else if (def->isa<BottomType>()) {
        } else if (auto fntype = def->isa<FnType>()) {
            json arg_types = json::array();
            for (auto arg : fntype->ops()) {
                arg_types.push_back(translate_def(arg));
            }
            result["args"] = arg_types;
        } else if (auto closuretype = def->isa<ClosureType>()) {
            json args = json::array();
            for (auto arg : closuretype->ops()) {
                args.push_back(translate_def(arg));
            }
            result["args"] = args;
        } else if (def->isa<FrameType>()) {
        } else if (def->isa<MemType>()) {
        } else if (auto structtype = def->isa<StructType>()) {
            json arg_names = json::array();
            for (size_t i = 0; i < structtype->num_ops(); ++i) {
                arg_names.push_back(structtype->op_name(i).str());
            }

            json args = json::array();
            for (size_t i = 0; i < structtype->num_ops(); ++i) {
                args.push_back(translate_def(structtype->op(i)));
            }

            result["struct_name"] = structtype->name().str();
            result["arg_names"] = arg_names;
            result["args"] = args;
        } else if (auto varianttype = def->isa<VariantType>()) {
            json arg_names = json::array();
            for (size_t i = 0; i < varianttype->num_ops(); ++i) {
                arg_names.push_back(varianttype->op_name(i).str());
            }
            json args = json::array();
            for (size_t i = 0; i < varianttype->num_ops(); ++i) {
                args.push_back(translate_def(varianttype->op(i)));
            }

            result["variant_name"] = varianttype->name().str();
            result["args"] = args;
            result["arg_names"] = arg_names;
        } else if (auto tupletype = def->isa<TupleType>()) {
            json args = json::array();
            for (size_t i = 0; i < tupletype->num_ops(); ++i) {
                args.push_back(translate_def(tupletype->op(i)));
            }

            result["args"] = args;
        } else if (auto prim = def->isa<PrimType>()) {
            result["length"] = prim->length();
            switch (prim->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimTypeTag::PrimType_##T: { result["tag"] = #T; break; }
#include <thorin/tables/primtypetable.h>
            }
        } else if (auto ptrtype = def->isa<PtrType>()) {
            auto pointee_type = translate_def(ptrtype->pointee());
            result["pointee"] = pointee_type;
            result["length"] = ptrtype->length();
            switch (ptrtype->addr_space()) {
                case AddrSpace::Generic:
                    //result["addrspace"] = "generic"; //Default
                    break;
                case AddrSpace::Global:
                    result["addrspace"] = "global";
                    break;
                case AddrSpace::Texture:
                    result["addrspace"] = "texture";
                    break;
                case AddrSpace::Shared:
                    result["addrspace"] = "shared";
                    break;
                case AddrSpace::Constant:
                    result["addrspace"] = "constant";
                    break;
                case AddrSpace::Private:
                    result["addrspace"] = "private";
                    break;
                case AddrSpace::Function:
                    result["addrspace"] = "function";
                    break;
                case AddrSpace::Push:
                    result["addrspace"] = "push";
                    break;
                case AddrSpace::Input:
                    result["addrspace"] = "input";
                    break;
                case AddrSpace::Output:
                    result["addrspace"] = "output";
                    break;
            }
        } else if (auto cont = def->isa_nom<Continuation>()) {
            if (cont->is_intrinsic()) {
                result["intrinsic"] = get_intrinsic_name(cont->intrinsic());
            }
            if (cont->filter() && !cont->filter()->empty())
                result["filter"] = translate_def(cont->filter());

            if (cont->intrinsic() == Intrinsic::Match) {
                size_t num_patterns = cont->num_params() - 3;
                auto variant_type = translate_def(cont->param(1)->type());
                result["variant_type"] = variant_type;
                result["num_patterns"] = num_patterns;
            }
            //TODO: Is this actually required for imported functions?
            json arg_names = json::array();
            for (auto arg : cont->params()) {
                arg_names.push_back(translate_def(arg));
            }
            result["arg_names"] = arg_names;
            switch (cont->cc()) {
                case CC::Thorin:
                    break;
                case CC::C:
                    result["cc"] = "C";
                    break;
                case CC::Device:
                    result["cc"] = "device";
                    break;
            }
            if(cont->has_body())
                result["body"] = translate_def(cont->body());
        } else if (auto app = def->isa<App>()) {
            result["callee"] = translate_def(app->callee());
            json args = json::array();
            for (auto arg : app->args()) {
                args.push_back(translate_def(arg));
            }
            result["args"] = args;
        } else if (auto lit = def->isa<PrimLit>()) {
            result["primtype_tag"] = lit->primtype_tag();
            switch (lit->primtype_tag()) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: { result["value"] = std::to_string(lit->value().get_##M()); break; }
#define THORIN_BOOL_TYPE(T, M) case PrimType_##T: { result["value"] = std::to_string(lit->value().get_##M()); break; }
#define THORIN_F_TYPE(T, M) case PrimType_##T: { result["value"] = std::to_string(lit->value().get_##M()); break; }
#include <thorin/tables/primtypetable.h>
                default:
                    assert(false && "not implemented");
            }
        } else if (def->isa<Top>()) {
            auto type = translate_def(def->type());
            result["const_type"] = type;
        } else if (def->isa<Bottom>()) {
            auto type = translate_def(def->type());
            result["const_type"] = type;
        } else if (auto param = def->isa<Param>()) {
            result["continuation"] = translate_def(param->continuation());
            result["index"] = param->index();
            auto name = param->continuation()->unique_name() + "." + std::to_string(param->index());
            return name;
        } else if (auto load = def->isa<Load>()) {
            json args = json::array();
            args.push_back(translate_def(load->mem()));
            args.push_back(translate_def(load->ptr()));
            result["args"] = args;
        } else if (auto store = def->isa<Store>()) {
            json args = json::array();
            args.push_back(translate_def(store->mem()));
            args.push_back(translate_def(store->ptr()));
            args.push_back(translate_def(store->val()));
            result["args"] = args;
        } else if (auto size_of = def->isa<SizeOf>()) {
            auto target_type = translate_def(size_of->of());
            result["target_type"] = target_type;
        } else if (auto align_of = def->isa<AlignOf>()) {
            auto target_type = translate_def(align_of->of());
            result["target_type"] = target_type;
        }  else if (auto cast = def->isa<Cast>()) {
            auto source = translate_def(cast->from());
            auto target_type = translate_def(cast->type());
            result["source"] = source;
            result["target_type"] = target_type;
        } else if (auto bitcast = def->isa<Bitcast>()) {
            auto source = translate_def(bitcast->from());
            auto target_type = translate_def(bitcast->type());
            result["source"] = source;
            result["target_type"] = target_type;
        } else if (auto indef_array = def->isa<IndefiniteArray>()) {
            auto dim = translate_def(indef_array->op(0));
            auto element_type = translate_def(indef_array->elem_type());
            result["elem_type"] = element_type;
            result["dim"] = dim;
        } else if (auto def_array = def->isa<DefiniteArray>()) {
            json args = json::array();
            for (auto arg : def_array->ops()) {
                args.push_back(translate_def(arg));
            }
            auto element_type = translate_def(def_array->elem_type());
            result["elem_type"] = element_type;
            result["args"] = args;
        } else if (auto lea = def->isa<LEA>()) {
            json args = json::array();
            args.push_back(translate_def(lea->ptr()));
            args.push_back(translate_def(lea->index()));
            result["args"] = args;
        } else if (auto extract = def->isa<Extract>()) {
            json args = json::array();
            args.push_back(translate_def(extract->agg()));
            args.push_back(translate_def(extract->index()));
            result["args"] = args;
        } else if (auto insert = def->isa<Insert>()) {
            json args = json::array();
            args.push_back(translate_def(insert->agg()));
            args.push_back(translate_def(insert->index()));
            args.push_back(translate_def(insert->value()));
            result["args"] = args;
        } else if (auto closure = def->isa<Closure>()) {
            json args = json::array();
            args.push_back(translate_def(closure->op(0)));
            args.push_back(translate_def(closure->op(1)));
            result["args"] = args;
        } else if (auto struct_agg = def->isa<StructAgg>()) {
            json args = json::array();
            for (auto arg : struct_agg->ops()) {
                args.push_back(translate_def(arg));
            }
            result["args"] = args;
        } else if (auto tuple = def->isa<Tuple>()) {
            json args = json::array();
            for (auto arg : tuple->ops()) {
                args.push_back(translate_def(arg));
            }
            result["args"] = args;
        } else if (auto vector = def->isa<Vector>()) {
            json args = json::array();
            for (auto arg : vector->ops()) {
                args.push_back(translate_def(arg));
            }
            result["args"] = args;
        } else if (auto filter = def->isa<Filter>()) {
            json args = json::array();
            for (auto arg : filter->ops()) {
                args.push_back(translate_def(arg));
            }
            result["args"] = args;
        } else if (auto arithop = def->isa<ArithOp>()) {
            auto op = arithop->op_name();
            json args = json::array();
            args.push_back(translate_def(arithop->lhs()));
            args.push_back(translate_def(arithop->rhs()));
            result["op"] = op;
            result["args"] = args;
        } else if (auto mathop = def->isa<MathOp>()) {
            auto op = mathop->op_name();
            json args = json::array();
            for (auto arg : mathop->ops()) {
                args.push_back(translate_def(arg));
            }
            result["op"] = op;
            result["args"] = args;
        } else if (auto select = def->isa<Select>()) {
            json args = json::array();
            args.push_back(translate_def(select->cond()));
            args.push_back(translate_def(select->tval()));
            args.push_back(translate_def(select->fval()));
            result["args"] = args;
        } else if (auto cmp = def->isa<Cmp>()) {
            auto op = cmp->op_name();
            json args = json::array();
            args.push_back(translate_def(cmp->lhs()));
            args.push_back(translate_def(cmp->rhs()));
            result["op"] = op;
            result["args"] = args;
        } else if (auto run = def->isa<Run>()) {
            auto target = translate_def(run->def());
            result["target"] = target;
        } else if (auto hlt = def->isa<Hlt>()) {
            auto target = translate_def(hlt->def());
            result["target"] = target;
        } else if (auto known = def->isa<Known>()) {
            auto def2 = translate_def(known->def());
            result["def"] = def2;
        } else if (auto enter = def->isa<Enter>()) {
            auto mem = translate_def(enter->mem());
            result["mem"] = mem;
        } else if (auto slot = def->isa<Slot>()) {
            auto frame = translate_def(slot->frame());
            auto target_type = translate_def(slot->alloced_type());
            result["frame"] = frame;
            result["target_type"] = target_type;
        } else if (auto alloc = def->isa<Alloc>()) {
            json args = json::array();
            args.push_back(translate_def(alloc->mem()));
            args.push_back(translate_def(alloc->extra()));
            auto target_type = translate_def(alloc->alloced_type());
            result["args"] = args;
            result["target_type"] = target_type;
        } else if (auto global = def->isa<Global>()) {
            bool is_mutable = global->is_mutable();
            bool is_externally_defined = global->is_imported();
            if (!global->init()->isa<Bottom>()) {
                auto init = translate_def(global->init());
                result["init"] = init;
            }
            result["mutable"] = is_mutable;
            if (is_externally_defined)
                result["external"] = global->name();
        } else if (auto variant = def->isa<Variant>()) {
            auto value = translate_def(variant->value());
            size_t index = variant->index();
            result["value"] = value;
            result["index"] = index;
        } else if (auto variant_extract = def->isa<VariantExtract>()) {
            auto value = translate_def(variant_extract->value());
            size_t index = variant_extract->index();
            result["value"] = value;
            result["index"] = index;
        } else if (auto variant_index = def->isa<VariantIndex>()) {
            auto value = translate_def(variant_index->op(0));
            result["value"] = value;
        } else if (auto assembly = def->isa<Assembly>()) {
            auto asm_type = translate_def(assembly->type());
            json inputs = json::array();
            inputs.push_back(translate_def(assembly->mem()));
            for (auto input : assembly->inputs()) {
                inputs.push_back(translate_def(input));
            }
            auto asm_template = assembly->asm_template();
            json out_constraints = json::array();
            for (auto constraint : assembly->output_constraints()) {
                out_constraints.push_back(constraint);
            }
            json in_constraints = json::array();
            for (auto constraint : assembly->input_constraints()) {
                in_constraints.push_back(constraint);
            }
            json clobbers = json::array();
            for (auto c : assembly->clobbers()) {
                clobbers.push_back(c);
            }

            result["asm_type"] = asm_type;
            result["inputs"] = inputs;
            result["asm_template"] = asm_template;

            result["output_constraints"] = out_constraints;
            result["input_constraints"] = in_constraints;
            result["clobbers"] = clobbers;
            switch (assembly->flags()) {
                case Assembly::Flags::NoFlag:
                    result["flags"] = "noflag";
                    break;
                case Assembly::Flags::HasSideEffects:
                    result["flags"] = "hassideeffects";
                    break;
                case Assembly::Flags::IsAlignStack:
                    result["flags"] = "isalignstack";
                    break;
                case Assembly::Flags::IsIntelDialect:
                    result["flags"] = "isinteldialect";
                    break;
            }
        } else if (def->isa_structural()) {
            json args = json::array();
            for (auto op : def->ops())
                args.push_back(translate_def(op));
            result["args"] = args;
        } else {
            def->dump();
            def->dump(2);
            std::cerr << "cannot be translated\n";
            THORIN_UNREACHABLE;
        }
        if (def->isa_nom())
            return expect_translated_def(def);
        return def_name(def);
    }

    std::string expect_translated_def(const Def* def) {
        auto it = known_defs.find(def);
        assert(it != known_defs.end());
        return it->second;
    }

    std::string translate_def_flat(const Def* root) {
        assert(root->isa_structural());
        std::stack<const Def*> rebuild_stack;
        std::queue<const Def*> todo;
        rebuild_stack.push(root);
        todo.push(root);

        while (!todo.empty()) {
            auto def = todo.front();
            todo.pop();
            if (known_defs.lookup(def)) continue;

            for (auto& op : def->ops()) {
                if (op->isa_structural()) {
                    rebuild_stack.push(def);
                    todo.push(op);
                } else {
                    translate_def(op);
                }
            }
        }

        while (!rebuild_stack.empty()) {
            auto def = pop(rebuild_stack);
            // this might happen due to nominal nodes being rebuilt immediately - not using the queue
            if (known_defs.lookup(def))
                continue;
            std::vector<std::string> ops;
            for (auto& op : def->ops())
                ops.push_back(expect_translated_def(op));
            json result = json::object();
            translate_def_with_ops(def, ops, result);
            auto name = def_name(def);
            defs[name] = result;
            known_defs[def] = name;
        }

        return expect_translated_def(root);
    }

    std::string translate_def(const Def* def) {
        auto it = known_defs.find(def);
        if (it != known_defs.end()) {
            return it->second;
        }

        std::optional<std::string> nominal_name;
        if (def->isa_nom()) {
            nominal_name = known_defs[def] = def_name(def);
        }

        std::vector<std::string> ops;
        for (auto &op: def->ops())
            ops.push_back(translate_def(op));

        json result = json::object();
        auto name = translate_def_with_ops(def, ops, result);
        defs[name] = result;
        if (def->isa_nom()) {
            assert(*nominal_name == name);
        } else {
            known_defs[def] = name;
        }
        return name;
    }
};

void CodeGen::emit_json(json& j) {
    j["module"] = world().name();
    if (target_triple != "")
        j["target_triple"] = target_triple;
    if (target_cpu != "")
        j["target_cpu"] = target_cpu;
    if (target_attr != "")
        j["target_attr"] = target_attr;

    DefTable def_table;

    json externals = json::array();
    for (auto& [_, external] : world().externals()) {
        externals.push_back(def_table.translate_def(external));
    }

    j["externals"] = externals;
    j["defs"] = def_table.defs;
}

void CodeGen::emit_stream(std::ostream& stream) {
    json j;

    emit_json(j);
    
    Stream s(stream);
    s << j.dump(2) << "\n";
}

}
