#pragma once

#include "model_types.h"

#include <string>

namespace pvrgpu::stub {

bool ConfigureDriverCommandOptions(Options *options, std::string *error);
int RunConfiguredModel(Options options);

}  // namespace pvrgpu::stub
