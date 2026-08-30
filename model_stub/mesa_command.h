#pragma once

#include "model_types.h"

#include <string>

namespace pvrgpu::stub {

bool LoadMesaPocCommand(const std::string &path, MesaPocCommand *command,
                        std::string *error);

} // namespace pvrgpu::stub
