#include "model_types.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>

namespace pvrgpu::stub {
namespace {

bool ParseUnsigned(const char* text, unsigned* value) {
  if (!text || !text[0])
    return false;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (!end || end[0] || parsed == 0 ||
      parsed > std::numeric_limits<unsigned>::max()) {
    return false;
  }
  *value = static_cast<unsigned>(parsed);
  return true;
}

bool ParseOnOff(const char* text, bool* value) {
  if (!text)
    return false;
  const std::string mode(text);
  if (mode == "off") {
    *value = false;
    return true;
  }
  if (mode == "on") {
    *value = true;
    return true;
  }
  return false;
}

bool ParseMemoryMode(const char *text, MemoryMode *mode) {
  if (!text || !mode)
    return false;
  const std::string value(text);
  if (value == "direct") {
    *mode = MemoryMode::kDirect;
    return true;
  }
  if (value == "bypass") {
    *mode = MemoryMode::kBypass;
    return true;
  }
  if (value == "cache") {
    *mode = MemoryMode::kCache;
    return true;
  }
  return false;
}

bool ParseTextureMemoryPath(const char *text, TextureMemoryPath *path) {
  if (!text || !path)
    return false;
  const std::string value(text);
  if (value == "short") {
    *path = TextureMemoryPath::kShort;
    return true;
  }
  if (value == "cached") {
    *path = TextureMemoryPath::kCached;
    return true;
  }
  return false;
}

bool ParseTextureLodMode(const char *text, bool *exact) {
  if (!text || !exact)
    return false;
  const std::string value(text);
  if (value == "llvmpipe") {
    *exact = false;
    return true;
  }
  if (value == "exact") {
    *exact = true;
    return true;
  }
  return false;
}

}  // namespace

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--frames N] [--width N] [--height N] [--case NAME]"
                   " [--outdir PATH] [--memory-mode direct|bypass|cache]"
                   " [--texture-memory-path short|cached]"
                   " [--texture-lod-mode llvmpipe|exact]"
                   " [--cache-bypass on|off]"
                   " [--driver-command PATH]\n"
                   "  --memory-mode cache   Run SLC/DRAM simulation (default)\n"
                   "  --memory-mode bypass  Bypass cache but retain DRAM timing\n"
                   "  --memory-mode direct  Direct DRAM backing access (fast)\n"
                   "  --texture-memory-path short   TPU to DRAM short path (default)\n"
                   "  --texture-memory-path cached TPU to TCU to SLC to DRAM\n"
                   "  --texture-lod-mode exact     Conformance/hardware LOD math\n"
                   "  --texture-lod-mode llvmpipe Capture/Play-compatible LOD math (default)\n"
                   "  --cache-bypass on|off Legacy alias for bypass|cache\n"
                   "  --driver-command PATH Ingest one pvrgpu.driver-command.v1 command\n";
      std::exit(0);
    }
    if (i + 1 >= argc) {
      std::cerr << "Missing value after " << arg << "\n";
      return false;
    }
    const char* value = argv[++i];
    if (arg == "--frames") {
      if (!ParseUnsigned(value, &options->frames))
        return false;
    } else if (arg == "--width") {
      if (!ParseUnsigned(value, &options->width))
        return false;
    } else if (arg == "--height") {
      if (!ParseUnsigned(value, &options->height))
        return false;
    } else if (arg == "--case") {
      options->test_case = value;
    } else if (arg == "--outdir") {
      options->output_dir = value;
    } else if (arg == "--driver-command") {
      options->driver_command_path = value;
    } else if (arg == "--memory-mode") {
      if (!ParseMemoryMode(value, &options->memory_mode)) {
        std::cerr << "Invalid value for --memory-mode: " << value
                  << " (expected direct, bypass, or cache)\n";
        return false;
      }
      options->cache_bypass = options->memory_mode == MemoryMode::kBypass;
    } else if (arg == "--texture-memory-path") {
      if (!ParseTextureMemoryPath(value, &options->texture_memory_path)) {
        std::cerr << "Invalid value for --texture-memory-path: " << value
                  << " (expected short or cached)\n";
        return false;
      }
    } else if (arg == "--texture-lod-mode") {
      if (!ParseTextureLodMode(value, &options->exact_texture_lod)) {
        std::cerr << "Invalid value for --texture-lod-mode: " << value
                  << " (expected llvmpipe or exact)\n";
        return false;
      }
    } else if (arg == "--cache-bypass") {
      if (!ParseOnOff(value, &options->cache_bypass)) {
        std::cerr << "Invalid value for --cache-bypass: " << value
                  << " (expected on or off)\n";
        return false;
      }
      options->memory_mode = options->cache_bypass ? MemoryMode::kBypass
                                                   : MemoryMode::kCache;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      return false;
    }
  }
  if (options->texture_memory_path == TextureMemoryPath::kCached &&
      options->memory_mode != MemoryMode::kCache) {
    std::cerr << "--texture-memory-path cached requires --memory-mode cache "
                 "and is incompatible with cache bypass\n";
    return false;
  }
  return true;
}

const char *MemoryModeName(MemoryMode mode) {
  switch (mode) {
  case MemoryMode::kDirect:
    return "direct";
  case MemoryMode::kBypass:
    return "bypass";
  case MemoryMode::kCache:
    return "cache";
  }
  return "invalid";
}

const char *TextureMemoryPathName(TextureMemoryPath path) {
  switch (path) {
  case TextureMemoryPath::kShort:
    return "short";
  case TextureMemoryPath::kCached:
    return "cached";
  }
  return "invalid";
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream output;
  for (const unsigned char c : value) {
    switch (c) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (c < 0x20) {
          output << "\\u00";
          static constexpr char kHex[] = "0123456789abcdef";
          output << kHex[(c >> 4) & 0xf] << kHex[c & 0xf];
        } else {
          output << c;
        }
    }
  }
  return output.str();
}

std::uint32_t WorkloadClass(const std::string& test_case) {
  if (test_case == "driver_clear_color")
    return 5;
  if (test_case == "driver_triangle_solid")
    return 6;
  if (test_case == "driver_indexed_quad")
    return 7;
  if (test_case == "driver_textured_triangles")
    return 8;
  if (test_case == "driver_pco_triangles")
    return 9;
  if (test_case.find("triangle_setup") != std::string::npos)
    return 1;
  if (test_case.find("attribute_fetch") != std::string::npos)
    return 2;
  if (test_case.find("varyings") != std::string::npos)
    return 3;
  if (test_case.find("tex") != std::string::npos)
    return 4;
  return 0;
}

std::ostream& operator<<(std::ostream& stream, const PoolHandle& handle) {
  return stream << handle.slot << ':' << handle.generation;
}

std::ostream& operator<<(std::ostream& stream, const PipelineTxn& txn) {
  return stream << "PipelineTxn(frame=" << txn.frame << ", handle="
                << txn.state << ')';
}

std::ostream& operator<<(std::ostream& stream, const MemoryTxn& txn) {
  return stream << "MemoryTxn(frame=" << txn.pipeline.frame
                << ", state=" << txn.pipeline.state
                << ", payload=" << txn.payload << ", address="
                << txn.address << ", bytes=" << txn.bytes
                << ", request_id=" << txn.request_id
                << ", response_route="
                << static_cast<unsigned>(txn.response_route)
                << ", batch_control="
                << static_cast<unsigned>(txn.batch_control) << ')';
}

std::ostream& operator<<(std::ostream& stream, const CounterTxn& txn) {
  return stream << "CounterTxn(frame=" << txn.frame << ')';
}

}  // namespace pvrgpu::stub
