#include "impala/type.h"

#include "anydsl2/world.h"

using namespace anydsl2;

namespace impala {

TypeTable::TypeTable()
    : types_()
#define IMPALA_TYPE(itype, atype) ,itype##_(unify(primtype(Token::Type_##itype)))
#include "impala/tokenlist.h"
    , type_error_(unify(new TypeError()))
    , noret_(unify(new NoRet()))
    , unit_(unify(new TupleType(ArrayRef<const Type*>(nullptr, 0))))
{}

const Type* TypeTable::unify_base(const Type* type) {
    auto i = types_.find(type);
    if (i != types_.end()) {
        delete type;
        return *i;
    }

    auto p = types_.insert(type);
    assert(p.second && "hash/equal broken");
    return type;
}

const PrimType* TypeTable::primtype(TokenKind kind) {
    switch (kind) {
#define IMPALA_TYPE(itype, atype) case Token::Type_##itype: return itype##_;
#include "impala/tokenlist.h"
        default: ANYDSL2_UNREACHABLE;
    }
}

bool Type::is_bool() const { 
    if (auto pt = isa<PrimType>()) 
        return pt->kind() == Token::Type_bool; 
    return false; 
}

bool Type::is_int() const {
    if (auto pt = isa<PrimType>()) {
        switch (pt->kind()) {
            case Token::Type_int8:
            case Token::Type_int16:
            case Token::Type_int32:
            case Token::Type_int64:
            case Token::Type_int:   return true;
            default:                return false;
        }
    }
    return false;
}

const Type* FnType::return_type() const {
    if (!empty()) {
        if (auto ret = elems().back()->isa<FnType>())
            return typetable_.tupletype(ret->elems());
    }

    return typetable_.noret();
}

//------------------------------------------------------------------------------

const anydsl2::Type* PrimType::convert(World& world) const {
    switch (kind()) {
#define IMPALA_TYPE(itype, atype) case Token::Type_##itype: return world.type_##atype();
#include "impala/tokenlist.h"
        default: ANYDSL2_UNREACHABLE;
    }
}

const anydsl2::Type* FnType::convert(World& world) const {
    Array<const anydsl2::Type*> elems(size() + 1);
    elems[0] = world.mem();
    for (size_t i = 1, e = elems.size(); i != e; ++i)
        elems[i] = elem(i-1)->convert(world);

    return world.pi(elems);
}

const anydsl2::Type* TupleType::convert(World& world) const {
    Array<const anydsl2::Type*> elems(size());
    for (size_t i = 0, e = elems.size(); i != e; ++i)
        elems[i] = elem(i)->convert(world);

    return world.sigma(elems);
}

} // namespace impala
