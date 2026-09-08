// SPDX-License-Identifier: MIT
#pragma once
// Optional functional-test output, NOT snapshot serialization. Call only after
// the engine verified completion in the actual replay GL context. No debug
// context, RenderDoc SaveTexture, shader conversion or hidden resolve is used.
#include "rdc_runner/sha256.h"

#include <array>
#include <cerrno>
#include <cstdint>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace pvrgpu::rdc {
namespace drawlist_color_detail {
inline void Require(bool value, const std::string &message)
{ if(!value) throw std::runtime_error("DrawList color readback: " + message); }
inline std::string Quote(const std::string &value)
{
  std::ostringstream out;
  out << '"';
  for(unsigned char c : value)
    if(c == '"' || c == '\\') out << '\\' << c;
    else if(c < 32) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                       << unsigned(c) << std::dec;
    else out << c;
  return out.str() + '"';
}
struct GL {
  unsigned (*error)() = nullptr;
  void (*get)(unsigned, int *) = nullptr;
  const unsigned char *(*string)(unsigned) = nullptr;
  const unsigned char *(*stringIndex)(unsigned, unsigned) = nullptr;
  void (*attachment)(unsigned, unsigned, unsigned, int *) = nullptr;
  void (*bindFramebuffer)(unsigned, unsigned) = nullptr;
  void (*bindTexture)(unsigned, unsigned) = nullptr;
  void (*bindBuffer)(unsigned, unsigned) = nullptr;
  void (*texLevel)(unsigned, int, unsigned, int *) = nullptr;
  void (*pack)(unsigned, int) = nullptr;
  void (*readBuffer)(unsigned) = nullptr;
  void (*readPixels)(int, int, int, int, unsigned, unsigned, void *) = nullptr;
  unsigned (*checkFramebuffer)(unsigned) = nullptr;
  explicit GL(void *library) {
    Require(library != nullptr, "missing explicit GLES library");
#define PVRGPU_COLOR_SYMBOL(member, name) \
    member = reinterpret_cast<decltype(member)>(dlsym(library, name)); \
    Require(member != nullptr, "missing GLES symbol " name)
    PVRGPU_COLOR_SYMBOL(error, "glGetError");
    PVRGPU_COLOR_SYMBOL(get, "glGetIntegerv");
    PVRGPU_COLOR_SYMBOL(string, "glGetString");
    PVRGPU_COLOR_SYMBOL(stringIndex, "glGetStringi");
    PVRGPU_COLOR_SYMBOL(attachment, "glGetFramebufferAttachmentParameteriv");
    PVRGPU_COLOR_SYMBOL(bindFramebuffer, "glBindFramebuffer");
    PVRGPU_COLOR_SYMBOL(bindTexture, "glBindTexture");
    PVRGPU_COLOR_SYMBOL(bindBuffer, "glBindBuffer");
    PVRGPU_COLOR_SYMBOL(texLevel, "glGetTexLevelParameteriv");
    PVRGPU_COLOR_SYMBOL(pack, "glPixelStorei");
    PVRGPU_COLOR_SYMBOL(readBuffer, "glReadBuffer");
    PVRGPU_COLOR_SYMBOL(readPixels, "glReadPixels");
    PVRGPU_COLOR_SYMBOL(checkFramebuffer, "glCheckFramebufferStatus");
#undef PVRGPU_COLOR_SYMBOL
  }
  int Integer(unsigned name) const { int result = 0; get(name, &result); return result; }
  int Attachment(unsigned name) const {
    int result = 0; attachment(0x8CA9, 0x8CE0, name, &result); return result;
  }
  void Check(const char *phase) const {
    const unsigned result = error();
    Require(result == 0, std::string(phase) + " GL error=" + std::to_string(result));
  }
};

struct Restore {
  GL &gl;
  int readFbo, readBuffer, drawReadBuffer = 0, pbo, texture;
  std::array<int, 4> pack;
  bool reverseAvailable = false, changedReadFbo = false, restored = false;
  int reverse = 0;
  static constexpr std::array<unsigned, 4> parameters{{0x0D05, 0x0D02, 0x0D03, 0x0D04}};
  explicit Restore(GL &api) : gl(api) {
    readFbo = gl.Integer(0x8CAA); readBuffer = gl.Integer(0x0C02);
    pbo = gl.Integer(0x88ED); texture = gl.Integer(0x8069);
    for(size_t i = 0; i < pack.size(); ++i) pack[i] = gl.Integer(parameters[i]);
    const int extensions = gl.Integer(0x821D);
    for(int i = 0; i < extensions; ++i) {
      const auto *extension = gl.stringIndex(0x1F03, unsigned(i));
      if(extension && std::string(reinterpret_cast<const char *>(extension)) ==
                          "GL_ANGLE_pack_reverse_row_order") reverseAvailable = true;
    }
    if(reverseAvailable) reverse = gl.Integer(0x93A4);
    gl.Check("capture state");
  }
  void Apply() noexcept {
    if(restored) return;
    for(size_t i = 0; i < pack.size(); ++i) gl.pack(parameters[i], pack[i]);
    if(reverseAvailable) gl.pack(0x93A4, reverse);
    gl.bindBuffer(0x88EB, unsigned(pbo));
    gl.bindTexture(0x0DE1, unsigned(texture));
    if(changedReadFbo) gl.readBuffer(unsigned(drawReadBuffer));
    gl.bindFramebuffer(0x8CA8, unsigned(readFbo));
    gl.readBuffer(unsigned(readBuffer));
    restored = true;
  }
  ~Restore() { Apply(); }
};

inline void ExclusiveWrite(const std::filesystem::path &path, const std::vector<uint8_t> &bytes)
{
  Require(path.is_absolute(), "output must be absolute");
  std::filesystem::path walk;
  for(const auto &part : path) {
    Require(part != "..", "output traversal is forbidden"); walk /= part;
    Require(!std::filesystem::is_symlink(std::filesystem::symlink_status(walk)),
            "output symlink is forbidden");
  }
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
  Require(fd >= 0, "cannot exclusively create output " + path.string());
  size_t done = 0;
  while(done < bytes.size()) {
    const ssize_t amount = write(fd, bytes.data() + done, bytes.size() - done);
    if(amount < 0 && errno == EINTR) continue;
    if(amount <= 0) { close(fd); throw std::runtime_error("Incomplete DrawList color output"); }
    done += size_t(amount);
  }
  const bool synced = fsync(fd) == 0;
  const bool closed = close(fd) == 0;
  Require(synced && closed, "could not synchronize color output");
}
} // namespace drawlist_color_detail

struct CompletedDrawColor {
  std::string path, sha256;
  uint32_t width = 0, height = 0, mip = 0;
  uint64_t size_bytes = 0;
  std::string format = "RGBA8", origin = "bottom-left", source = "completed-replay-color0";
  std::string ToJson() const {
    using drawlist_color_detail::Quote;
    std::ostringstream out;
    out << "{\"path\":" << Quote(path) << ",\"sha256\":" << Quote(sha256)
        << ",\"width\":" << width << ",\"height\":" << height
        << ",\"size_bytes\":" << size_bytes << ",\"mip\":" << mip
        << ",\"format\":" << Quote(format) << ",\"origin\":" << Quote(origin)
        << ",\"source\":" << Quote(source) << '}';
    return out.str();
  }
};

inline CompletedDrawColor ReadCompletedDrawColor(void *explicitGlesLibrary,
                                                const std::filesystem::path &output)
{
  using namespace drawlist_color_detail;
  GL gl(explicitGlesLibrary);
  gl.Check("entry (pre-existing errors are not swallowed)");
  Require(gl.string(0x1F02) != nullptr, "no current GL context");
  const int drawFbo = gl.Integer(0x8CA6);
  Require(drawFbo != 0, "default/missing draw FBO has no verified texture extent");
  Require(gl.checkFramebuffer(0x8CA9) == 0x8CD5, "draw framebuffer is incomplete");
  Require(gl.Integer(0x80A8) == 0 && gl.Integer(0x80A9) == 0, "MSAA readback is unsupported");
  Require(gl.Integer(0x8825) == 0x8CE0, "draw output 0 is not COLOR_ATTACHMENT0");
  Require(gl.Attachment(0x8CD0) == 0x1702, "color0 must be a texture");
  const int name = gl.Attachment(0x8CD1), mip = gl.Attachment(0x8CD2);
  Require(name != 0 && mip >= 0 && mip < 32, "invalid color0 texture/mip");
  Require(gl.Attachment(0x8CD3) == 0 && gl.Attachment(0x8CD4) == 0 &&
          gl.Attachment(0x8DA7) == 0, "cube/layered color readback is unsupported");
  gl.Check("attachment metadata");
  Restore restore(gl);
  gl.bindTexture(0x0DE1, unsigned(name));
  gl.Check("color0 must be a 2D texture");
  int width = 0, height = 0, format = 0;
  gl.texLevel(0x0DE1, mip, 0x1000, &width);
  gl.texLevel(0x0DE1, mip, 0x1001, &height);
  gl.texLevel(0x0DE1, mip, 0x1003, &format);
  gl.Check("texture extent/format");
  Require(format == 0x8058, "v1 verification codec requires RGBA8 storage");
  Require(width > 0 && height > 0 && size_t(width) <=
              std::numeric_limits<size_t>::max() / 4 / size_t(height), "invalid color extent");
  std::vector<uint8_t> pixels(size_t(width) * size_t(height) * 4);
  gl.bindFramebuffer(0x8CA8, unsigned(drawFbo));
  restore.drawReadBuffer = gl.Integer(0x0C02);
  restore.changedReadFbo = true;
  gl.readBuffer(0x8CE0);
  gl.bindBuffer(0x88EB, 0);
  for(size_t i = 0; i < restore.pack.size(); ++i) gl.pack(Restore::parameters[i], i == 0 ? 1 : 0);
  if(restore.reverseAvailable) gl.pack(0x93A4, 0);
  gl.Check("readback setup");
  gl.readPixels(0, 0, width, height, 0x1908, 0x1401, pixels.data());
  const unsigned readError = gl.error();
  restore.Apply();
  gl.Check("state restoration");
  Require(readError == 0, "glReadPixels failed: " + std::to_string(readError));
  ExclusiveWrite(output, pixels);
  CompletedDrawColor result;
  result.path = output.string(); result.width = uint32_t(width); result.height = uint32_t(height);
  result.mip = uint32_t(mip); result.size_bytes = pixels.size();
  std::string error;
  Require(Sha256File(output, &result.sha256, &error), error);
  return result;
}
} // namespace pvrgpu::rdc
