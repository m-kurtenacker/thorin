#include "json.h"

#include <string>

namespace thorin {

namespace json {

struct JSonImporter {
    World& world_;
    json& j_;

    std::unordered_map<std::string, const Def*> defs_;

    JSonImporter(World& world, json& j) : world_(world), j_(j) {}

    Array<const Def*> get_ops(const json& args) {
        Array<const Def*> defs(args.size());
        for (size_t i = 0; i < args.size(); i++) {
            defs[i] = get_def(args[i]);
        }
        return defs;
    }

    Array<const Type*> get_types(const json& args) {
        Array<const Type*> defs(args.size());
        for (size_t i = 0; i < args.size(); i++) {
            defs[i] = get_def(args[i])->as<Type>();
        }
        return defs;
    }

    const Def* parse_def(std::string& name, json j) {
        std::string tag_name = j["tag"];

        const Type* type = nullptr;
        if (!j["def_type"].empty())
            type = get_def(j["def_type"])->as<Type>();

        NodeTag tag = str2tag(tag_name);
        switch (tag) {
#define THORIN_GLUE(pre, next)
#define THORIN_PRIMTYPE(T)    case Node_PrimType_##T: return world_.prim_type(PrimType_##T, j["length"]);
#define THORIN_NODE(n, abbr)
#define THORIN_ARITHOP(n)     case Node_##n: return world_.arithop(ArithOp_##n, get_def(j["args"][0]), get_def(j["args"][1]));
#define THORIN_CMP(n)         case Node_##n: return world_.cmp(Cmp_##n, get_def(j["args"][0]), get_def(j["args"][1]));
#define THORIN_MATHOP(n)      case Node_##n: return world_.mathop(MathOp_##n, get_ops(j["args"]));
#include "thorin/tables/allnodes.h"
            case Node_Star: return world_.star();
            case Node_BotType: return world_.bottom_type();
            case Node_FrameType: return world_.frame_type();
            case Node_MemType: return world_.mem_type();
            case Node_FnType: return world_.fn_type(get_types(j["args"]));
            case Node_ReturnType: return world_.return_type(get_types(j["args"]));
            case Node_ClosureType: return world_.closure_type(get_types(j["args"]));
            case Node_PtrType: {
                AddrSpace as = AddrSpace::Generic;
                if (j["addrspace"] != nullptr) {
                    if (j["addrspace"] == "global")
                        as = AddrSpace::Global;
                    else
                        throw std::runtime_error("TODO");
                }
                return world_.ptr_type(get_def(j["pointee"])->as<Type>(), 1, as);
            }
            case Node_TupleType: return world_.tuple_type(get_types(j["args"]));
            case Node_StructType: {
                auto tys = get_types(j["args"]);
                auto s = world_.struct_type(Symbol(static_cast<std::string>(j["struct_name"])), tys.size());
                defs_[name] = s;
                for (size_t i = 0; i < tys.size(); i++)
                    s->set_op(i, tys[i]);
                return s;
            }
            case Node_IndefiniteArrayType: return world_.indefinite_array_type(get_def(j["args"][0])->as<Type>());
            case Node_DefiniteArrayType: return world_.definite_array_type(get_def(j["args"][0])->as<Type>(), j["length"]);
            case Node_Continuation: {
                Continuation::Attributes attributes;
                if (j["cc"] == "C")
                    attributes.cc == CC::C;
                if (j["cc"] == "device")
                    attributes.cc == CC::Device;
                Continuation* cont = world_.continuation(type->as<FnType>(), attributes);
                defs_[name] = cont;
                //for (auto p : j["arg_names"]) {
                //    get_def(p)->as<Param>()->set_name()
                //}
                if (!j["body"].empty())
                    cont->set_body(get_def(j["body"])->as<App>());
                if (!j["filter"].empty())
                    cont->set_filter(get_def(j["filter"])->as<Filter>());
                cont->set_name(name);
                return cont;
            }
            case Node_Global: {
                const Def* init = nullptr;
                if (j["init"] != nullptr)
                    init = get_def(j["init"]);
                return world_.global(init, j["mutable"], j["external"] != nullptr, {});
            }
            case Node_App: return world_.app(get_def(j["callee"]), get_ops(j["args"]));
            case Node_ReturnPoint: return world_.return_point(get_def(j["args"][0])->as_nom<Continuation>());
            case Node_Run: return world_.run(get_def(j["target"]));
            case Node_Hlt: return world_.hlt(get_def(j["target"]));
            case Node_Known: return world_.known(get_def(j["def"]));
            case Node_Enter: return world_.enter(get_def(j["mem"]));
            case Node_Filter: return world_.filter(get_ops(j["args"]));
            case Node_Param: {
                Continuation* cont = get_def(j["continuation"])->as_nom<Continuation>();
                return cont->param(j["index"]);
            }
            case Node_Literal: {
                PrimTypeTag tag = j["primtype_tag"];
                std::string value = j["value"];
                switch (tag) {
                    case PrimType_bool: return world_.literal_bool(value == "true", {});
                    case PrimType_pu8: return world_.literal_pu8(std::strtoull(value.c_str(), nullptr, 10), {});
                    case PrimType_qu8: return world_.literal_qu8(std::strtoull(value.c_str(), nullptr, 10), {});
                    case PrimType_pu16: return world_.literal_pu16(std::strtoull(value.c_str(), nullptr, 10), {});
                    case PrimType_qu16: return world_.literal_qu16(std::strtoull(value.c_str(), nullptr, 10), {});
                    case PrimType_pu32: return world_.literal_pu32(std::strtoull(value.c_str(), nullptr, 10), {});
                    case PrimType_qu32: return world_.literal_qu32(std::strtoull(value.c_str(), nullptr, 10), {});
                    case PrimType_pu64: return world_.literal_pu64(std::strtoull(value.c_str(), nullptr, 10), {});
                    case PrimType_qu64: return world_.literal_qu64(std::strtoull(value.c_str(), nullptr, 10), {});
                    case PrimType_ps8: return world_.literal_ps8(std::strtol(value.c_str(), nullptr, 10), {});
                    case PrimType_qs8: return world_.literal_qs8(std::strtol(value.c_str(), nullptr, 10), {});
                    case PrimType_ps16: return world_.literal_ps16(std::strtol(value.c_str(), nullptr, 10), {});
                    case PrimType_qs16: return world_.literal_qs16(std::strtol(value.c_str(), nullptr, 10), {});
                    case PrimType_ps32: return world_.literal_ps32(std::strtol(value.c_str(), nullptr, 10), {});
                    case PrimType_qs32: return world_.literal_qs32(std::strtol(value.c_str(), nullptr, 10), {});
                    case PrimType_ps64: return world_.literal_ps64(std::strtoll(value.c_str(), nullptr, 10), {});
                    case PrimType_qs64: return world_.literal_qs64(std::strtoll(value.c_str(), nullptr, 10), {});
                    case PrimType_pf16: return world_.literal_pf16(half(std::strtof(value.c_str(), nullptr)), {});
                    case PrimType_qf16: return world_.literal_qf16(half(std::strtof(value.c_str(), nullptr)), {});
                    case PrimType_pf32: return world_.literal_pf32(std::strtof(value.c_str(), nullptr), {});
                    case PrimType_qf32: return world_.literal_qf32(std::strtof(value.c_str(), nullptr), {});
                    case PrimType_pf64: return world_.literal_pf64(std::strtod(value.c_str(), nullptr), {});
                    case PrimType_qf64: return world_.literal_qf64(std::strtod(value.c_str(), nullptr), {});
                }
            }
            case Node_SizeOf: return world_.size_of(get_def(j["target_type"])->as<Type>());
            case Node_AlignOf: return world_.align_of(get_def(j["target_type"])->as<Type>());
            case Node_Bitcast: return world_.bitcast(get_def(j["target_type"])->as<Type>(), get_def(j["source"]));
            case Node_Cast: return world_.cast(get_def(j["target_type"])->as<Type>(), get_def(j["source"]));
            case Node_DefiniteArray: return world_.definite_array(get_ops(j["args"]));
            case Node_Tuple: return world_.tuple(get_ops(j["args"]));
            case Node_StructAgg: return world_.struct_agg(type->as<StructType>(), get_ops(j["args"]));
            default: throw std::runtime_error("unknown tag '" + tag_name + "'");
        }
    }

    const Def* get_def(std::string name) {
        auto found = defs_.find(name);
        if (found != defs_.end())
            return found->second;
        auto def = parse_def(name, j_["defs"][name]);
        defs_[name] = def;
        return def;
    }

    void spawn_externals() {
        for (auto& e : j_["externals"]) {
            auto def = get_def(e);
            world_.make_external(const_cast<Def*>(def));
        }
    }
};

void load_defs(World& world, json& j) {
    JSonImporter importer(world, j);
    importer.spawn_externals();
}

}

}
