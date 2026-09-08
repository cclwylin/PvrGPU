#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace pvrgpu::rdc {

using GLDebugCallback = void (*)(unsigned, unsigned, unsigned, unsigned, int,
                                 const char *, const void *);

struct GLDebugApi {
  void *(*current)() = nullptr;
  unsigned (*error)() = nullptr;
  void (*get)(unsigned, int *) = nullptr;
  void (*getPointer)(unsigned, void **) = nullptr;
  unsigned char (*isEnabled)(unsigned) = nullptr;
  void (*enable)(unsigned) = nullptr;
  void (*disable)(unsigned) = nullptr;
  void (*callback)(GLDebugCallback, const void *) = nullptr;
  void (*control)(unsigned, unsigned, unsigned, int, const unsigned *, unsigned char) = nullptr;
  void (*pushGroup)(unsigned, unsigned, int, const char *) = nullptr;
  void (*popGroup)() = nullptr;
};

// The capture must not be allowed to change error-observation state midway
// through the interval and restore it just before the final state check.
// Pinned RenderDoc does not serialize callback/control calls today; reject
// them if a future/imported capture does expose one instead of assuming that.
inline void ValidateDebugCaptureCall(const std::string &name, bool has_cap,
                                     std::uint64_t cap, std::uint64_t &groups) {
  const auto contains = [&](const char *token) { return name.find(token) != std::string::npos; };
  if (contains("DebugMessageCallback") || contains("DebugMessageControl") ||
      contains("MakeContextCurrent") || contains("ImplicitThreadSwitch"))
    throw std::runtime_error("Cannot guarantee API error observation across captured " + name);
  if (name == "glDisable" || name == "glDisablei" || name == "glDisableiEXT" ||
      name == "glDisableIndexedEXT" || name == "glDisableiOES" || name == "glDisableiNV") {
    if (!has_cap || cap == 0x92E0 || cap == 0x8242)
      throw std::runtime_error("Capture can disable synchronous API error observation: " + name);
  }
  if (contains("PushDebugGroup") || name == "glPushGroupMarkerEXT") {
    if (groups == UINT64_MAX) throw std::runtime_error("Capture debug group count overflow");
    ++groups;
  }
  if (contains("PopDebugGroup") || name == "glPopGroupMarkerEXT") {
    if (!groups) throw std::runtime_error("Capture debug-group pop can escape API error observation scope");
    --groups;
  }
}

class GLDebugScope {
  struct CallbackState {
    GLDebugCallback previous = nullptr;
    const void *user = nullptr;
    std::uint64_t errors = 0;
    unsigned first_id = 0;
    std::array<char, 512> first_message{};
  };
 public:
  explicit GLDebugScope(const GLDebugApi &api) : api_(api) {
    Require(api_.current && api_.error && api_.get && api_.getPointer && api_.isEnabled &&
            api_.enable && api_.disable && api_.callback && api_.control &&
            api_.pushGroup && api_.popGroup, "Synchronous GL debug API is unavailable");
    context_ = api_.current();
    Require(context_ != nullptr, "API error scope has no replay EGL context");
    void *callback = nullptr, *user = nullptr;
    api_.getPointer(kCallbackFunction, &callback);
    api_.getPointer(kCallbackUser, &user);
    previous_callback_ = reinterpret_cast<GLDebugCallback>(callback);
    previous_user_ = user;
    state_->previous = previous_callback_;
    state_->user = previous_user_;
    previous_enabled_ = api_.isEnabled(kOutput) != 0;
    previous_synchronous_ = api_.isEnabled(kSynchronous) != 0;
    api_.get(kGroupDepth, &previous_depth_);
    int maximum = 0;
    api_.get(kMaximumGroupDepth, &maximum);
    CheckGL("API debug state capture");
    Require(previous_depth_ >= 1 && previous_depth_ < maximum,
            "No debug-group capacity for independent API error observation");
    try {
      // Message filters are per debug group. A push inherits them; changing
      // only this group's filters and popping later restores the original
      // per-source/type/severity/ID settings without guessing their values.
      api_.pushGroup(0x824A, 0, -1, "PvrGPU native replay API validation");
      CheckGL("API debug filter scope push");
      group_pushed_ = true;
      api_.callback(&Receive, state_.get());
      callback_installed_ = true;
      api_.enable(kOutput);
      api_.enable(kSynchronous);
      api_.control(0x1100, 0x1100, 0x1100, 0, nullptr, 1);
      Check("API debug scope setup");
    } catch (...) { RestoreNoThrow(); throw; }
  }
  GLDebugScope(const GLDebugScope &) = delete;
  GLDebugScope &operator=(const GLDebugScope &) = delete;
  ~GLDebugScope() { RestoreNoThrow(); }

  void Check(const char *phase) const {
    Require(api_.current() == context_, "API debug observer changed EGL context");
    void *callback = nullptr, *user = nullptr;
    api_.getPointer(kCallbackFunction, &callback);
    api_.getPointer(kCallbackUser, &user);
    int depth = 0;
    api_.get(kGroupDepth, &depth);
    Require(callback == reinterpret_cast<void *>(&Receive) && user == state_.get() &&
            api_.isEnabled(kOutput) && api_.isEnabled(kSynchronous) &&
            depth == previous_depth_ + 1,
            "Synchronous API debug callback/state was replaced or disabled");
    CheckGL(phase);
    Require(state_->errors == 0, std::string(phase) + ": GL debug error " +
            std::to_string(state_->first_id) + " (" + std::to_string(state_->errors) +
            " observed): " + state_->first_message.data());
  }

  // Explicit success cleanup is checked; exception cleanup never masks the
  // original failure. A callback is restored before its userdata can die.
  void Finish() {
    Check("Completed replay and attachment readback");
    RestoreNoThrow();
    CheckGL("API debug state restoration");
    void *callback = nullptr, *user = nullptr;
    api_.getPointer(kCallbackFunction, &callback);
    api_.getPointer(kCallbackUser, &user);
    int depth = 0;
    api_.get(kGroupDepth, &depth);
    Require(callback == reinterpret_cast<void *>(previous_callback_) && user == previous_user_ &&
            bool(api_.isEnabled(kOutput)) == previous_enabled_ &&
            bool(api_.isEnabled(kSynchronous)) == previous_synchronous_ &&
            depth == previous_depth_, "Could not restore original GL debug state");
    CheckGL("Restored API debug state verification");
  }

 private:
  static constexpr unsigned kOutput = 0x92E0, kSynchronous = 0x8242;
  static constexpr unsigned kCallbackFunction = 0x8244, kCallbackUser = 0x8245;
  static constexpr unsigned kGroupDepth = 0x826D, kMaximumGroupDepth = 0x826C;
  static void Require(bool condition, const std::string &why) {
    if (!condition) throw std::runtime_error(why);
  }
  void CheckGL(const char *phase) const {
    const unsigned error = api_.error();
    Require(error == 0, std::string(phase) + ": GL error " + std::to_string(error));
  }
  static void Receive(unsigned source, unsigned type, unsigned id, unsigned severity,
                       int length, const char *message, const void *user) noexcept {
    auto *self = const_cast<CallbackState *>(static_cast<const CallbackState *>(user));
    // Keep a bounded, allocation-free receipt even if the driver cleared its
    // sticky GL error flag internally. Callback exceptions cannot cross C ABI.
    if (type == 0x824C || severity == 0x9146) { // ERROR or HIGH severity
      if (self->errors != UINT64_MAX) ++self->errors;
      if (self->errors == 1) {
        self->first_id = id;
        const std::size_t available = self->first_message.size() - 1;
        std::size_t count = 0;
        if (message) {
          while (count < available && (length < 0 || count < static_cast<unsigned>(length)) && message[count]) ++count;
          std::memcpy(self->first_message.data(), message, count);
        }
        self->first_message[count] = 0;
      }
    }
    if (self->previous) {
      try { self->previous(source, type, id, severity, length, message, self->user); }
      catch (...) {
        // A prior callback is application/library code, not a reason to lose
        // our observer or terminate across the GL callback boundary.
        if (self->errors != UINT64_MAX) ++self->errors;
      }
    }
  }
  void RestoreNoThrow() noexcept {
    if (!group_pushed_ && !callback_installed_) return;
    // Same-context ownership is required by Check. Never mutate another
    // context while unwinding a replay that changed contexts unexpectedly.
    if (api_.current() != context_) {
      // The run already fails closed. Keep only the tiny callback receipt
      // alive until process exit: shutdown may make the original context
      // current again, so freeing its still-installed userdata would be UAF.
      if (callback_installed_) (void)state_.release();
      callback_installed_ = group_pushed_ = false;
      return;
    }
    if (callback_installed_) {
      api_.callback(previous_callback_, previous_user_);
      callback_installed_ = false;
      (previous_enabled_ ? api_.enable : api_.disable)(kOutput);
      (previous_synchronous_ ? api_.enable : api_.disable)(kSynchronous);
    }
    if (group_pushed_) {
      int depth = 0;
      api_.get(kGroupDepth, &depth);
      if (depth == previous_depth_ + 1) api_.popGroup();
      group_pushed_ = false;
    }
  }
  GLDebugApi api_;
  void *context_ = nullptr;
  GLDebugCallback previous_callback_ = nullptr;
  const void *previous_user_ = nullptr;
  bool previous_enabled_ = false, previous_synchronous_ = false;
  bool group_pushed_ = false, callback_installed_ = false;
  int previous_depth_ = 0;
  std::unique_ptr<CallbackState> state_ = std::make_unique<CallbackState>();
};

}  // namespace pvrgpu::rdc
