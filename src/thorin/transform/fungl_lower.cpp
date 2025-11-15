#include "fungl_lower.h"

#include <spirv/unified1/spirv.hpp>

#include "rewrite.h"

#include "spirv/unified1/spirv.h"

namespace thorin {

struct FunGLLower : Rewriter {
    bool is_device_;
    FunGLLower(World& src, World& dst, bool is_device) : Rewriter(src, dst), is_device_(is_device) {

    }

    const ExternType* make_sampler_type() {
        std::vector<const Def*> ops;
        ops.push_back(dst().literal_pu32(SpvOpTypeSampler, { "opcode" }));
        ops.push_back(dst().tuple({}));
        auto nty = dst().extern_type("spirv.type", ops.size(), {});
        for (size_t i = 0; i < ops.size(); i++)
            nty->set_op(i, ops[i]);
        return nty;
    }

    const ExternType* make_image_type() {
        std::vector<const Def*> ops;
        ops.push_back(dst().literal_pu32(SpvOpTypeImage, { "opcode" }));
        ops.push_back(dst().tuple({
            dst().top(dst().unit_type()),
            dst().literal_pu32(spv::Dim2D, { "dim" }),
            dst().literal_pu32(0, { "depth"}),
            dst().literal_pu32(0, { "arrayed"}),
            dst().literal_pu32(0, { "multisampled"}),
            dst().literal_pu32(1, { "sampled"}),
            dst().literal_pu32(spv::ImageFormatUnknown, { "image format"}),
        }));
        ops.push_back(dst().type_pf32());
        auto nty = dst().extern_type("spirv.type", ops.size(), {});
        for (size_t i = 0; i < ops.size(); i++)
            nty->set_op(i, ops[i]);
        return nty;
    }

    const ExternType* make_sampled_image_type(const Type* image) {
        std::vector<const Def*> ops;
        ops.push_back(dst().literal_pu32(SpvOpTypeSampledImage, { "opcode" }));
        ops.push_back(dst().tuple({
            dst().top(dst().unit_type()),
        }));
        ops.push_back(image);
        auto nty = dst().extern_type("spirv.type", ops.size(), {});
        for (size_t i = 0; i < ops.size(); i++)
            nty->set_op(i, ops[i]);
        return nty;
    }

    const Def* rewrite(const Def* odef) override {
        if (auto ety = odef->isa<ExternType>()) {
            if (ety->name() == "fungl.image2d") {
                if (is_device_) {
                    auto it = make_image_type();
                    return make_sampled_image_type(it);
                } else {
                    auto struct_t = dst().struct_type("Image2D", 0);
                    return dst().ptr_type(struct_t);
                }
            }
        }
        return Rewriter::rewrite(odef);
    }
};

void fungl_lower(std::unique_ptr<World>& world, bool is_device) {
    auto fresh_world = std::make_unique<World>(*world);
    FunGLLower importer(*world, *fresh_world, is_device);
    for (auto&& [_, def] : world->externals()) {
        importer.instantiate(def);
    }
    std::swap(world, fresh_world);
}

}
