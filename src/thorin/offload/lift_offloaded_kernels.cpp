#include "lift_offloaded_kernels.h"

#include "thorin/transform/rewrite.h"
#include "thorin/transform/lift.h"

namespace thorin {

struct LiftOffloadedKernel {
    World& src_, &dst_;
    Offload &offload_;
    ScopesForest forest_;
    struct R;
    std::unique_ptr<R> root_;

    LiftOffloadedKernel(World& src, World& dst, Offload& offload) : src_(src), dst_(dst), offload_(offload), forest_(src) {
        root_ = std::make_unique<R>(*this, nullptr, nullptr);
    }

    struct R : Rewriter {
        LiftOffloadedKernel& converter_;
        Scope* scope_;
        R* parent_;
        std::vector<std::unique_ptr<R>> children_;

        DefSet fvs_;
        DefSet rematerialize_;

        const Def* rewrite(const thorin::Def *odef) override;

        R(LiftOffloadedKernel& converter, Scope* scope, R* parent) : Rewriter(converter.src_, converter.dst_), converter_(converter), scope_(scope), parent_(parent) {
            if (scope) {
                spillable_free_defs(scope->entry(), converter_.forest_, fvs_, rematerialize_);
            }
        }
    };
};

const Def* LiftOffloadedKernel::R::rewrite(const thorin::Def* odef) {
    if (parent_) {
        if (!scope_->contains(odef) && !rematerialize_.contains(odef)) {
            return parent_->instantiate(odef);
        }
    }

    if (auto app = odef->isa<App>()) {
        auto ncallee = instantiate(app->callee());
        auto ncont = ncallee->isa_nom<Continuation>();

        std::vector<const Def*> nargs;
        nargs.resize(app->num_args());

        auto ret_param_i = app->callee_type()->ret_param_index();
        for (size_t i = 0; i < app->num_args(); i++) {
            auto oarg = app->arg(i);
            // do not convert accelerator kernels
            if (ncont && ncont->is_accelerator()) {
                if (oarg->type()->isa<FnType>() && !oarg->type()->isa<ReturnType>())
                    continue;
            }
            nargs[i] = instantiate(oarg);
        }

        if (ncont) {
            if (ncont->is_accelerator()) {
                struct S : OffloadSite {
                    const Def*& host_mem_;
                    std::vector<const Def*> lifted_args;
                    std::vector<const Type*> env_types;

                    void add_host_arg(const Def* def) override {
                        lifted_args.push_back(def);
                        env_types.push_back(def->type());
                    }

                    const Def*& host_mem() override {
                        return host_mem_;
                    }

                    S(Rewriter* r, const Def*& mem) : OffloadSite(r), host_mem_(mem) {}
                };
                assert(is_mem(nargs[0]));
                S site = S(this, nargs[0]);

                auto backend = converter_.offload_.find_backend_for_intrinsic(ncont->intrinsic());

                struct K : OffloadSite::Kernel {
                    size_t idx;
                    R* body_rewriter;
                    Array<const Def*> args;
                    const Def*& mem_;

                    const Def*& find_mem() {
                        for (size_t i = 0; i < args.size(); i++) {
                            if (is_mem(args[i]))
                                return args[i];
                        }
                        throw std::runtime_error("no mem");
                    }

                    const Def*& mem() override {
                        return mem_;
                    }

                    K(Continuation* old, Continuation* w, size_t i, R* r) : Kernel(old, w), idx(i), body_rewriter(r), args(w->params_as_defs()), mem_(find_mem()) {}

                    void insert_mapping(const Def* old, const Def* def) override {
                        body_rewriter->insert(old, def);
                    }

                    const Def* get_mapping(const Def* old) override {
                        return body_rewriter->instantiate(old);
                    }
                };

                // find the kernels and make a K object for each
                for (size_t i = 0; i < ncont->num_params(); i++) {
                    if (!nargs[i]) {
                        auto old_kernel = app->arg(i)->as_nom<Continuation>();
                        auto& scope = converter_.forest_.get_scope(old_kernel);
                        children_.emplace_back(std::make_unique<R>(converter_, &scope, this));

                        Continuation* wrapper = dst().continuation(dst().fn_type(instantiate(old_kernel->type())->as<FnType>()->types()));
                        site.kernels_.emplace_back(std::make_unique<K>(old_kernel, wrapper, i, children_.back().get()));
                    }
                }

                // iterate over the kernels and register params for them
                for (auto& kernel : site.kernels_) {
                    auto& k = *reinterpret_cast<K*>(&*kernel);
                    for (auto ofv : k.body_rewriter->fvs_) {
                        if (backend)
                            backend->lower_env_param(ofv, site);
                        else
                            default_lower_env_param(ofv, site);
                    }
                }

                // we add them once to the call
                // we're going to change the type of the accelerator ofc
                std::vector<const Type*> nintrinsic_types;
                for (auto t : ncont->type()->copy_types())
                    nintrinsic_types.push_back(t);
                for (auto env : site.lifted_args) {
                    nintrinsic_types.push_back(env->type());
                    nargs.push_back(env);
                }

                for (auto& kernel : site.kernels_) {
                    auto& k = *reinterpret_cast<K*>(&*kernel);
                    k.wrapper->jump(k.body_rewriter->instantiate(k.old_kernel), k.args);

                    nargs[k.idx] = k.wrapper;
                    nintrinsic_types[k.idx] = k.wrapper->type();
                }

                auto nintrinsic = dst().continuation(dst().fn_type(nintrinsic_types), ncont->attributes(),ncont->debug());
                ncallee = nintrinsic;
            }
        }
        return dst().app(ncallee, nargs, app->debug());
    }
    return Rewriter::rewrite(odef);
}

void lower_offloaded_kernels(std::unique_ptr<World>& src, Offload& offload) {
    auto dst = std::make_unique<World>(*src);
    LiftOffloadedKernel converter(*src, *dst, offload);

    for (auto& [_, external] : src->externals()) {
        auto next = converter.root_->instantiate(external);
        //dst->make_external(const_cast<Def *>(next));
    }

    src.swap(dst);
}

}
