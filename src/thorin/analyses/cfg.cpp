#include "thorin/analyses/cfg.h"
#include "thorin/analyses/scope.h"
#include "thorin/analyses/domtree.h"
#include "thorin/analyses/looptree.h"

namespace thorin {

CFG::CFG(const Scope& scope) 
    : scope_(scope)
    //, nodes_(scope.size() + 1)                      // one extra alloc for virtual exit
    , nodes_(scope.size())
{
    for (size_t i = 0, e = size(); i != e; ++i)
        nodes_[i] = new CFGNode(scope.rpo(i));
    //nodes_.back() = new CFGNode(nullptr);           // virtual exit

    //for (auto n : nodes_.slice_num_from_end(1)) {  // skip virtual exit
    for (auto n : nodes_) {
        for (auto succ : n->lambda()->succs()) {
            if (scope.contains(succ))
                link(n, nodes_[sid(succ)]);
        }
    }
}

size_t CFG::sid(Lambda* lambda) const { 
    return lambda == nullptr ? nodes_.size()-1 : lambda->find_scope(&scope())->sid; 
}

struct FlowVal {
    LambdaSet lambdas;
    bool top = false;
    bool join(const FlowVal& other) {
        bool result = false;
        for (auto l : other.lambdas)
            result |= this->lambdas.insert(l).second;
        return result;
    }
};

void CFG::cfa() {
    DefMap<FlowVal> param2fv;

    // init
    for (auto lambda : scope()) {
        if (!lambda->empty()) {
            if (auto to = lambda->to()->isa_lambda()) {
                for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                    if (auto arg = lambda->arg(i)->isa_lambda())
                        param2fv[to->param(i)].lambdas.insert(arg);
                }
            }
        }
    }

    // keep iterating to collect param flow infos until things are stable
    bool todo;
    do {
        todo = false;
        for (auto lambda : scope()) {
            if (auto to = lambda->to()->isa_lambda()) {
                for (size_t i = 0, e = lambda->num_args(); i != e; ++i) {
                    if (auto arg = lambda->arg(i)->isa<Param>())
                        todo |= param2fv[to->param(i)].join(param2fv[arg]);
                }
            }
        }
    } while (todo);
}

const CFGView<true>*  CFG::forwards()  const { return lazy_init(this, forwards_); }
const CFGView<false>* CFG::backwards() const { return lazy_init(this, backwards_); }
const DomTree* CFG::domtree() const { return forwards()->domtree(); }
const PostDomTree* CFG::postdomtree() const { return backwards()->domtree(); }
const LoopTree* CFG::looptree() const { return looptree_ ? looptree_ : looptree_ = new LoopTree(*forwards()); }

template<bool forward>
const DomTreeBase<forward>* CFGView<forward>::domtree() const { return lazy_init(this, domtree_); }

}
