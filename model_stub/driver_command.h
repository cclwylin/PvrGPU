#pragma once

#include "model_types.h"

#include <string>

namespace pvrgpu::stub {

bool LoadDriverCommand(const std::string &path, DriverCommand *command,
                       std::string *error);

} // namespace pvrgpu::stub
