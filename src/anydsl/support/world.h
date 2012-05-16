#ifndef ANYDSL_SUPPORT_WORLD_H
#define ANYDSL_SUPPORT_WORLD_H

#include <cassert>
#include <string>

#include <boost/unordered_map.hpp>

#include "anydsl/air/enums.h"
#include "anydsl/util/box.h"
#include "anydsl/util/autoptr.h"

namespace anydsl {

class ArithOp;
class Def;
class Pi;
class PrimConst;
class PrimType;
class Sigma;

//------------------------------------------------------------------------------

typedef boost::unordered_multimap<uint64_t, Def*> Defs;
typedef Defs::iterator DefIter;
typedef boost::unordered_multimap<uint64_t, Pi*> Pis;
typedef boost::unordered_multimap<uint64_t, Sigma*> Sigmas;
typedef std::vector<Sigma*> NamedSigmas;

//------------------------------------------------------------------------------

/**
 * This class manages the following things for the whole program:
 *  - Type unification:
 *      There exists only one unique type for PrimType%s, Pi%s and \em unnamed Sigma%s.
 *      These types are hashed into internal maps for fast access.
 *      The getters just calculate a hash and lookup the type, if it is already present, or create a new one otherwise.
 *      There also exists the concept of \em named \p Sigma%s to allow for recursive types.
 *      These types are \em not unified, i.e., each instance is by definition a different type;
 *      thus, two different pointers of the same named sigma are considered different types.
 *  - PrimOp unification:
 *      This is a built-in mechanism for the following things:
 *      - common subexpression elimination
 *      - constant folding 
 *      - copy propagation
 *      - dead code elimination
 *      - canonicalization of expressions
 *      - several local optimizations
 *      PrimOp%s do not explicitly belong to a Lambda.
 *      Instead they either implicitly belong to a Lambda 
 *      when they (possibly via multiple steps) depend on an Lambda's Param or they are dead. 
 *      Use \p cleanup to remove dead code.
 *  - Lambda%s are register here in order to not have dangling pointers 
 *  and to perform unreachable code elimination.
 *  The aforementioned \p cleanup will also delete these lambdas.
 *
 *  You can create several worlds. 
 *  All worlds are independent from each other.
 *  This is particular useful for multi-threading.
 */
class World {
public:

    World();
    ~World();

    ArithOp* createArithOp(ArithOpKind arithOpKind,
                           Def* ldef, Def* rdef, 
                           const std::string& ldebug = "", 
                           const std::string& rdebug = "", 
                           const std::string&  debug = "");

#define ANYDSL_U_TYPE(T) PrimType* type_##T() const { return T##_; }
#define ANYDSL_F_TYPE(T) PrimType* type_##T() const { return T##_; }
#include "anydsl/tables/primtypetable.h"

    PrimType* type(PrimTypeKind kind) const { 
        size_t i = kind - Begin_PrimType;
        assert(0 <= i && i < (size_t) Num_PrimTypes); 
        return primTypes_[i];
    }

    template<class T>
    PrimConst* constant(T value) { 
        return constant(Type2PrimTypeKind<T>::kind, Box(value));
    }
    PrimConst* constant(PrimTypeKind kind, Box value);

    template<class T>
    const Sigma* sigma(T begin, T end);

    /// Creates a fresh named sigma.
    Sigma* getNamedSigma(const std::string& name = "");

    const Pi* emptyPi() const { return emptyPi_; }
    const Sigma* unit() const { return unit_; }

    /// Performs dead code and unreachable code elimination.
    void cleanup();

private:

    Defs defs_;
    Pis pis_;
    Sigmas sigmas_;
    NamedSigmas namedSigmas_;

    AutoPtr<Pi> emptyPi_; ///< pi().
    AutoPtr<Sigma> unit_; ///< sigma().

    union {
        struct {
#define ANYDSL_U_TYPE(T) PrimType* T##_;
#define ANYDSL_F_TYPE(T) PrimType* T##_;
#include "anydsl/tables/primtypetable.h"
        };

        PrimType* primTypes_[Num_PrimTypes];
    };
};

} // namespace anydsl

#endif // ANYDSL_SUPPORT_WORLD_H
