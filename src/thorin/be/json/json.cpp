#include "json.h"

namespace thorin::json {

class TypeTable {
public:
    json type_table = json::array();

    TypeMap<std::string> known_types;

    std::string translate_type (const Type * type) {
        auto it = known_types.find(type);
        if (it != known_types.end()) {
            return it->second;
        }

        json result;
        if (type->isa<MemType>()) {
            result["name"] = "mem_t";
            result["type"] = "mem";
        } else if (auto prim = type->isa<PrimType>()) {
            result["name"] = "_" + std::to_string(type_table.size());
            result["length"] = prim->length();
            result["type"] = "prim";
            switch (prim->primtype_tag()) {
#define THORIN_ALL_TYPE(T, M) case PrimTypeTag::PrimType_##T: { result["tag"] = #T; break; }
#include <thorin/tables/primtypetable.h>
            }
        } else if (auto fntype = type->isa<FnType>()) {
            json arg_types = json::array();
            for (auto arg : fntype->ops()) {
                arg_types.push_back(translate_type(arg));
            }

            result["type"] = "fn";
            result["name"] = "_" + std::to_string(type_table.size());
            result["args"] = arg_types;
        } else if (auto ptrtype = type->isa<PtrType>()) {
            auto pointee_type = translate_type(ptrtype->pointee());

            result["type"] = "ptr";
            result["args"] = { pointee_type };
            result["name"] = pointee_type + "_p";
            result["length"] = ptrtype->length();
        } else if (auto arr = type->isa<IndefiniteArrayType>()) {
            auto elem_type = translate_type(arr->elem_type());

            result["type"] = "indef_array";
            result["args"] = { elem_type };
            result["name"] = elem_type + "_iarr";
        } else if (auto arr = type->isa<DefiniteArrayType>()) {
            auto elem_type = translate_type(arr->elem_type());

            result["type"] = "def_array";
            result["args"] = { elem_type };
            result["length"] = arr->dim();
            result["name"] = elem_type + "_iarr";
        } else {
            std::cerr << "type cannot be translated\n";
            type->dump();
            THORIN_UNREACHABLE;
        }
        known_types[type] = result["name"];
        type_table.push_back(result);
        return result["name"];
    }
};

class DefTable {
public:
    DefTable(TypeTable& type_table) : type_table_(type_table) {}

    json decl_table = json::array();
    json def_table = json::array();
    TypeTable& type_table_;

    DefMap<std::string> known_defs;

    std::string translate_def (const Def * def, std::string expected_name = "") {
        auto it = known_defs.find(def);
        if (it != known_defs.end()) {
            return it->second;
        }

        json result;
        if (auto cont = def->isa<Continuation>()) {
            if (cont->is_intrinsic()) {
                assert(cont->intrinsic() == Intrinsic::Branch && "TODO: anything else is unsupported RN");

                result["name"] = "branch";
                result["type"] = "continuation";
                result["intrinsic"] = "branch";
            } else if (cont->is_imported()) {
                auto name = cont->name();
                auto type = type_table_.translate_type(def->type());

                result["name"] = name;
                result["type"] = "continuation";
                result["fn_type"] = type;
                result["imported"] = true;
                result["external"] = cont->is_external();
            } else {
                assert(cont->has_body());

                auto type = type_table_.translate_type(def->type());
                json arg_names = json::array();
                for (auto arg : cont->params()) {
                    arg_names.push_back(translate_def(arg));
                }

                auto name = expected_name != "" ? expected_name : "_cont_" + std::to_string(decl_table.size());

                json forward_decl;
                forward_decl["name"] = name;
                forward_decl["type"] = "continuation";
                forward_decl["fn_type"] = type;
                forward_decl["arg_names"] = arg_names;
                forward_decl["external"] = cont->is_external();
                decl_table.push_back(forward_decl);

                known_defs[def] = name;

                auto app = cont->body();
                auto target = translate_def(app->callee());
                json args = json::array();
                for (auto arg : app->args()) {
                    args.push_back(translate_def(arg));
                }

                result["name"] = name;
                result["type"] = "continuation";
                result["app"] = {
                    {"target", target},
                    {"args", args}
                };
            }
        } else if (auto lit = def->isa<PrimLit>()) {
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());
            auto type = type_table_.translate_type(lit->type());

            result["name"] = name;
            result["type"] = "const";
            result["const_type"] = type;
            //result["value"] = lit->value().get_s32(); //TODO: this looks wrong. What I get should depend on the lit type.
            switch (lit->primtype_tag()) {
#define THORIN_I_TYPE(T, M) case PrimType_##T: { result["value"] = lit->value().get_##M(); break; }
#define THORIN_BOOL_TYPE(T, M) case PrimType_##T: { result["value"] = lit->value().get_##M(); break; }
#define THORIN_F_TYPE(T, M) case PrimType_##T: { result["value"] = (double)lit->value().get_##M(); break; }
#include <thorin/tables/primtypetable.h>
            default:
                assert(false && "not implemented");
            }
        } else if (def->isa<Top>()) {
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());
            auto type = type_table_.translate_type(def->type());

            result["name"] = name;
            result["type"] = "top";
            result["const_type"] = type;
        } else if (def->isa<Bottom>()) {
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());
            auto type = type_table_.translate_type(def->type());

            result["name"] = name;
            result["type"] = "bottom";
            result["const_type"] = type;
        } else if (auto param = def->isa<Param>()) {
            auto name = expected_name != "" ? expected_name : param->continuation()->unique_name() + "." + std::to_string(param->index());
            known_defs[def] = name;
            return name;
        } else if (auto load = def->isa<Load>()) {
            json args = json::array();
            args.push_back(translate_def(load->mem()));
            args.push_back(translate_def(load->ptr()));
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "load";
            result["args"] = args;
        } else if (auto store = def->isa<Store>()) {
            json args = json::array();
            args.push_back(translate_def(store->mem()));
            args.push_back(translate_def(store->ptr()));
            args.push_back(translate_def(store->val()));
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "store";
            result["args"] = args;
        } else if (auto size_of = def->isa<SizeOf>()) {
            auto target_type = type_table_.translate_type(size_of->of());
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "sizeof";
            result["target_type"] = target_type;
        } else if (auto align_of = def->isa<AlignOf>()) {
            auto target_type = type_table_.translate_type(align_of->of());
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "alignof";
            result["target_type"] = target_type;
        } else if (auto cast = def->isa<Cast>()) {
            auto source = translate_def(cast->from());
            auto target_type = type_table_.translate_type(cast->type());
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "cast";
            result["source"] = source;
            result["target_type"] = target_type;
         } else if (auto bitcast = def->isa<Bitcast>()) {
            auto source = translate_def(bitcast->from());
            auto target_type = type_table_.translate_type(bitcast->type());
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "bitcast";
            result["source"] = source;
            result["target_type"] = target_type;
        } else if (auto indef_array = def->isa<IndefiniteArray>()) {
            auto dim = translate_def(indef_array->op(0));
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());
            auto element_type = type_table_.translate_type(indef_array->elem_type());

            result["name"] = name;
            result["type"] = "indef_array";
            result["elem_type"] = element_type;
            result["dim"] = dim;
        } else if (auto def_array = def->isa<DefiniteArray>()) {
            json args = json::array();
            for (auto arg : def_array->ops()) {
                args.push_back(translate_def(arg));
            }

            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());
            auto element_type = type_table_.translate_type(def_array->elem_type());

            result["name"] = name;
            result["type"] = "def_array";
            result["elem_type"] = element_type;
            result["args"] = args;
        } else if (auto lea = def->isa<LEA>()) {
            json args = json::array();
            args.push_back(translate_def(lea->ptr()));
            args.push_back(translate_def(lea->index()));
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "lea";
            result["args"] = args;
        } else if (auto extract = def->isa<Extract>()) {
            json args = json::array();
            args.push_back(translate_def(extract->agg()));
            args.push_back(translate_def(extract->index()));
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "extract";
            result["args"] = args;
        } else if (auto insert = def->isa<Insert>()) {
            json args = json::array();
            args.push_back(translate_def(insert->agg()));
            args.push_back(translate_def(insert->index()));
            args.push_back(translate_def(insert->value()));
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "insert";
            result["args"] = args;
        } else if (auto closure = def->isa<Closure>()) {
            json args = json::array();
            args.push_back(translate_def(closure->op(0)));
            args.push_back(translate_def(closure->op(1)));
            auto closure_type = type_table_.translate_type(closure->type());
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "closure";
            result["args"] = args;
            result["closure_type"] = closure_type;
        } else if (auto struct_agg = def->isa<StructAgg>()) {
            json args = json::array();
            for (auto arg : struct_agg->ops()) {
                args.push_back(translate_def(arg));
            }
            auto struct_type = type_table_.translate_type(struct_agg->type());
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "struct";
            result["args"] = args;
            result["struct_type"] = struct_type;
        } else if (auto tuple = def->isa<Tuple>()) {
            json args = json::array();
            for (auto arg : tuple->ops()) {
                args.push_back(translate_def(arg));
            }
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "tuple";
            result["args"] = args;
        } else if (auto vector = def->isa<Vector>()) {
            json args = json::array();
            for (auto arg : vector->ops()) {
                args.push_back(translate_def(arg));
            }
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "vector";
            result["args"] = args;
        } else if (auto filter = def->isa<Filter>()) {
            json args = json::array();
            for (auto arg : filter->ops()) {
                args.push_back(translate_def(arg));
            }
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "filter";
            result["args"] = args;
        } else if (auto arithop = def->isa<ArithOp>()) {
            auto op = arithop->op_name();
            json args = json::array();
            args.push_back(translate_def(arithop->lhs()));
            args.push_back(translate_def(arithop->rhs()));
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "arithop";
            result["op"] = op;
            result["args"] = args;
        } else if (auto select = def->isa<Select>()) {
            json args = json::array();
            args.push_back(translate_def(select->cond()));
            args.push_back(translate_def(select->tval()));
            args.push_back(translate_def(select->fval()));
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "select";
            result["args"] = args;
        } else if (auto cmp = def->isa<Cmp>()) {
            auto op = cmp->op_name();
            json args = json::array();
            args.push_back(translate_def(cmp->lhs()));
            args.push_back(translate_def(cmp->rhs()));
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "cmp";
            result["op"] = op;
            result["args"] = args;
        } else if (auto run = def->isa<Run>()) {
            auto target = translate_def(run->def());
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "run";
            result["target"] = target;
        } else if (auto hlt = def->isa<Hlt>()) {
            auto target = translate_def(hlt->def());
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "hlt";
            result["target"] = target;
        } else if (auto known = def->isa<Known>()) {
            auto def = translate_def(known->def());
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "known";
            result["def"] = def;
        } else if (auto enter = def->isa<Enter>()) {
            auto mem = translate_def(enter->mem());
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "enter";
            result["mem"] = mem;
        } else if (auto slot = def->isa<Slot>()) {
            auto frame = translate_def(slot->frame());
            auto target_type = type_table_.translate_type(slot->alloced_type());
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "slot";
            result["frame"] = frame;
            result["target_type"] = target_type;
        } else if (auto alloc = def->isa<Alloc>()) {
            json args = json::array();
            args.push_back(translate_def(alloc->mem()));
            args.push_back(translate_def(alloc->extra()));
            auto target_type = type_table_.translate_type(alloc->alloced_type());
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "alloc";
            result["args"] = args;
            result["target_type"] = target_type;
        } else if (auto global = def->isa<Global>()) {
            auto init = translate_def(global->init());
            bool is_mutable = global->is_mutable();
            auto name = expected_name != "" ? expected_name : "_" + std::to_string(def_table.size());

            result["name"] = name;
            result["type"] = "global";
            result["init"] = init;
            result["mutable"] = is_mutable;
        } else {
            def->dump();
            def->dump(2);
            std::cerr << "cannot be translated\n";
            THORIN_UNREACHABLE;
        }
        known_defs[def] = result["name"];
        def_table.push_back(result);
        return result["name"];
    }
};

void CodeGen::emit_stream(std::ostream& stream) {
    json j;

    j["module"] = world().name();

    TypeTable type_table;
    DefTable def_table(type_table);

    for (auto external : world().externals()) {
        const Continuation* continuation = external.second;
        auto expected_name = continuation->name();
        def_table.translate_def(continuation, expected_name);
    }

    j["type_table"] = type_table.type_table;
    j["defs"] = def_table.decl_table;
    for (auto it : def_table.def_table)
        j["defs"] += it;
    
    Stream s(stream);
    s << j.dump(2) << "\n";
}

}
