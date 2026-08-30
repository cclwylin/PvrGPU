#ifndef GLBENCH_MESA_MINI_GFLAGS_H_
#define GLBENCH_MESA_MINI_GFLAGS_H_

#include <cstdint>
#include <string>

// The Apple/Mesa adapter owns command-line parsing. These compatibility
// macros are only needed by a few unmodified GLBench source files.
#define DEFINE_bool(name, value, description) bool FLAGS_##name = value
#define DEFINE_double(name, value, description) double FLAGS_##name = value
#define DEFINE_int32(name, value, description) int32_t FLAGS_##name = value
#define DEFINE_string(name, value, description) std::string FLAGS_##name = value

#define DECLARE_bool(name) extern bool FLAGS_##name
#define DECLARE_double(name) extern double FLAGS_##name
#define DECLARE_int32(name) extern int32_t FLAGS_##name
#define DECLARE_string(name) extern std::string FLAGS_##name

#endif  // GLBENCH_MESA_MINI_GFLAGS_H_
