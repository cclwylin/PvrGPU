// Apple/Mesa platform adapter for the official ChromeOS GLBench workloads.

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "glinterface.h"
#include "main.h"

GLint g_width = WINDOW_WIDTH;
GLint g_height = WINDOW_HEIGHT;

std::unique_ptr<GLInterface> g_main_gl_interface;

namespace gl {
#define F(fun, type) type fun = nullptr;
LIST_PROC_FUNCTIONS(F)
#undef F
}  // namespace gl

namespace {

EGLContext ToEGLContext(const GLContext context) {
  return reinterpret_cast<EGLContext>(context);
}

GLContext FromEGLContext(const EGLContext context) {
  return reinterpret_cast<GLContext>(context);
}

class SurfacelessEGLInterface : public GLInterface {
 public:
  SurfacelessEGLInterface()
      : display_(EGL_NO_DISPLAY),
        config_(nullptr),
        surface_(EGL_NO_SURFACE),
        context_(nullptr) {}

  ~SurfacelessEGLInterface() override {
    if (display_ == EGL_NO_DISPLAY)
      return;
    eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (surface_ != EGL_NO_SURFACE)
      eglDestroySurface(display_, surface_);
    eglTerminate(display_);
  }

  bool Init() override {
    if (!InitOnce())
      return false;

    context_ = CreateContext();
    if (!context_)
      return false;
    if (!MakeCurrent(context_))
      return false;

#define F(fun, type) fun = reinterpret_cast<type>(eglGetProcAddress(#fun));
    LIST_PROC_FUNCTIONS(F)
#undef F
    return true;
  }

  void Cleanup() override {
    if (!context_)
      return;
    eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    DeleteContext(context_);
    context_ = nullptr;
  }

  void SwapBuffers() override { eglSwapBuffers(display_, surface_); }

  bool SwapInterval(int interval) override {
    return eglSwapInterval(display_, interval) == EGL_TRUE;
  }

  void CheckError() override {
    EGLint error = eglGetError();
    if (error != EGL_SUCCESS)
      std::fprintf(stderr, "GLBench EGL error: 0x%04x\n", error);
  }

  bool MakeCurrent(const GLContext& context) override {
    return eglMakeCurrent(display_, surface_, surface_, ToEGLContext(context)) ==
           EGL_TRUE;
  }

  const GLContext CreateContext() override {
    const EGLint attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    EGLContext context =
        eglCreateContext(display_, config_, EGL_NO_CONTEXT, attributes);
    if (context == EGL_NO_CONTEXT) {
      CheckError();
      return nullptr;
    }
    return FromEGLContext(context);
  }

  void DeleteContext(const GLContext& context) override {
    eglDestroyContext(display_, ToEGLContext(context));
  }

  const GLContext& GetMainContext() override { return context_; }

 private:
  bool InitOnce() {
    if (surface_ != EGL_NO_SURFACE)
      return true;

    PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
        reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
            eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (get_platform_display) {
      display_ = get_platform_display(EGL_PLATFORM_SURFACELESS_MESA, nullptr,
                                      nullptr);
    }
    if (display_ == EGL_NO_DISPLAY)
      display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display_ == EGL_NO_DISPLAY || !eglInitialize(display_, nullptr, nullptr)) {
      CheckError();
      return false;
    }
    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
      CheckError();
      return false;
    }

    const EGLint config_attributes[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES2_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE,
        8, EGL_ALPHA_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_STENCIL_SIZE, 8,
        EGL_NONE};
    EGLint config_count = 0;
    if (!eglChooseConfig(display_, config_attributes, &config_, 1,
                         &config_count) ||
        config_count != 1) {
      CheckError();
      return false;
    }

    const EGLint surface_attributes[] = {EGL_WIDTH, g_width, EGL_HEIGHT,
                                         g_height, EGL_NONE};
    surface_ = eglCreatePbufferSurface(display_, config_, surface_attributes);
    if (surface_ == EGL_NO_SURFACE) {
      CheckError();
      return false;
    }
    return true;
  }

  EGLDisplay display_;
  EGLConfig config_;
  EGLSurface surface_;
  GLContext context_;
};

}  // namespace

GLInterface* GLInterface::Create() {
  return new SurfacelessEGLInterface;
}
