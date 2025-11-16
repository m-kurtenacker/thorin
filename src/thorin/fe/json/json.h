#ifndef THORIN_FE_JSON_H
#define THORIN_FE_JSON_H

#include "thorin/world.h"
#include <nlohmann/json.hpp>

namespace thorin {

namespace json {

using json = nlohmann::json;

void load_defs(World&, json&);

}

}

#endif
