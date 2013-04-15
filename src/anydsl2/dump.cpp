#include <boost/typeof/typeof.hpp>

#include "anydsl2/lambda.h"
#include "anydsl2/literal.h"
#include "anydsl2/primop.h"
#include "anydsl2/printer.h"
#include "anydsl2/type.h"
#include "anydsl2/world.h"
#include "anydsl2/analyses/domtree.h"
#include "anydsl2/analyses/scope.h"
#include "anydsl2/analyses/placement.h"
#include "anydsl2/analyses/rootlambdas.h"
#include "anydsl2/util/for_all.h"

namespace anydsl2 {

/*
 * Types
 */

Printer& CompoundType::print_inner(Printer& p) const { ANYDSL2_DUMP_COMMA_LIST(p, elems()); return p; }
Printer& Frame::print(Printer& p) const { p << "frame"; return p; }
Printer& Mem::print(Printer& p) const { p << "mem"; return p; }
Printer& Pi::print(Printer& p) const { p << "pi("; print_inner(p); p << ')'; return p; }
Printer& Ptr::print(Printer& p) const { ref()->print(p); p << '*'; return p; }

Printer& PrimType::print(Printer& p) const {
	switch (primtype_kind()) {
#define ANYDSL2_UF_TYPE(T) case Node_PrimType_##T: p << #T; return p;
#include "anydsl2/tables/primtypetable.h"
	default: ANYDSL2_UNREACHABLE;
	}
}

Printer& Sigma::print(Printer& p) const {
    // TODO cycles
	p << "sigma(";
	print_inner(p);
    p << ")";
    return p;
}

Printer& Generic::print(Printer& p) const {
    if (!name.empty())
        p << name;
    else
        p << '_' << index();
    return p;
}

Printer& Opaque::print(Printer& p) const {
    p << "opaque(";
    for_all (f, flags()) p << f << " ";
    for_all (t,elems())  p << t << " ";
    p << ")";
    return p;
}

void Type::dump() const { Printer p(std::cout, false); print(p); }
void Def::dump() const { Printer p(std::cout, false); print(p); }
std::ostream& operator << (std::ostream& o, const anydsl2::Def* def) { Printer p(o, false); def->print(p); return p.o; }
Printer& Def::print(Printer& p) const { p.print_name(this); return p; }

Printer& Lambda::print_head(Printer& p) const {
	p.print_name(this);
    p << "(";
    ANYDSL2_DUMP_COMMA_LIST(p, params());
	p << ") : ";
    type()->print(p);
    if (attr().is_extern())
        p << " extern ";
    p.up();

    return p;
}

Printer& Lambda::print_jump(Printer& p) const {
    if (!empty()) {
        to()->print(p);
        p << "(";
        ANYDSL2_DUMP_COMMA_LIST(p, args());
        p  << ")";
    }
    p.down();

    return p;
}

//------------------------------------------------------------------------------

const char* PrimOp::op_name() const {
    switch (kind()) {
#define ANYDSL2_AIR_NODE(op) case Node_##op: return #op;
#include "anydsl2/tables/nodetable.h"
        default: ANYDSL2_UNREACHABLE;
    }
}

const char* ArithOp::op_name() const {
    switch (kind()) {
#define ANYDSL2_ARITHOP(op) case ArithOp_##op: return #op;
#include "anydsl2/tables/arithoptable.h"
        default: ANYDSL2_UNREACHABLE;
    }
}

const char* RelOp::op_name() const {
    switch (kind()) {
#define ANYDSL2_RELOP(op) case RelOp_##op: return #op;
#include "anydsl2/tables/reloptable.h"
        default: ANYDSL2_UNREACHABLE;
    }
}

const char* ConvOp::op_name() const {
    switch (kind()) {
#define ANYDSL2_CONVOP(op) case ConvOp_##op: return #op;
#include "anydsl2/tables/convoptable.h"
        default: ANYDSL2_UNREACHABLE;
    }
}

Printer& PrimOp::print(Printer& p) const {
    if (const PrimLit* primlit = this->isa<PrimLit>()) {
        switch (primlit->primtype_kind()) {
#define ANYDSL2_UF_TYPE(T) case PrimType_##T: p.o << primlit->box().get_##T(); break;
#include "anydsl2/tables/primtypetable.h"
            default: ANYDSL2_UNREACHABLE; break;
        }

        p << " : ";
        type()->print(p);
    } else if (is_const()) {
        p.print_name(this); 
        p << ' ';
        type()->print(p);
        p << ' ';

        if (size_t num = size()) {
            for (size_t i = 0; i != num-1; ++i) {
                op(i)->print(p);
                p << ", ";
            }
            op(num-1)->print(p);
        }
    } else
        p.print_name(this);

    return p;
}

Printer& PrimOp::print_assignment(Printer& p) const {
    p.print_name(this);
    p << " = " << op_name() << " : ";
    type()->print(p);
    p << ' ';

    if (size_t num = size()) {
        for (size_t i = 0; i != num-1; ++i) {
            op(i)->print(p);
            p << ", ";
        }
        op(num-1)->print(p);
    }
    p.newline();

    return p;
}

void World::dump(bool fancy) {
    Printer p(std::cout, fancy);

    for_all (root, find_root_lambdas(*this)) {
        Scope scope(root);
        Places places = place(scope);

        for_all (lambda, scope.rpo()) {
            int depth = fancy ? scope.domtree().depth(lambda) : 0;
            p.indent += depth;
            p.newline();
            lambda->print_head(p);

            for_all (op, places[lambda->sid()])
                op->print_assignment(p);

            lambda->print_jump(p);
            p.indent -= depth;
        }
    }
    p.newline();
}


} // namespace anydsl2
