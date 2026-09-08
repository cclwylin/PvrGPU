// SPDX-License-Identifier: MIT
// Three ordinary ordered draws in ONE complete capture; never suffix recapture.
// Linked to the explicitly selected RenderDoc EGL/GLES interception library.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include "api/app/renderdoc_app.h"
#include "drawlist-color-readback.h"

#include <array>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
static void check(bool condition, const std::string &message)
{ if(!condition) throw std::runtime_error(message); }
static void checkGL(const char *phase)
{ const auto error = glGetError(); check(error == GL_NO_ERROR, std::string(phase) + " GL error=" + std::to_string(error)); }
static GLuint shader(GLenum stage, const char *source)
{
  const GLuint id = glCreateShader(stage);
  glShaderSource(id, 1, &source, nullptr); glCompileShader(id);
  GLint compiled = 0; glGetShaderiv(id, GL_COMPILE_STATUS, &compiled);
  if(!compiled) { char log[8192] = {}; glGetShaderInfoLog(id, sizeof(log), nullptr, log); throw std::runtime_error(log); }
  return id;
}
static void positions(std::array<float, 24> &vertices, int draw, bool depth)
{
  constexpr float xy[12] = {-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1};
  for(unsigned vertex = 0; vertex < 6; ++vertex) {
    const float x = xy[2 * vertex];
    vertices[4 * vertex] = x; vertices[4 * vertex + 1] = xy[2 * vertex + 1];
    vertices[4 * vertex + 2] = !depth ? 0.f : draw == 0 ? .8f*x : draw == 1 ? 0.f : -1.f;
    vertices[4 * vertex + 3] = 1;
  }
}

int main(int argc, char **argv)
{
  try {
    check(argc == 4, "Usage: checkpoint-capture EXISTING_OUTPUT_DIRECTORY depth(0|1) hidden-mip(0|1)");
    const fs::path out = fs::canonical(argv[1]);
    const bool depth = std::string(argv[2]) == "1";
    check(depth || std::string(argv[2]) == "0", "depth must be 0 or 1");
    const bool hiddenMip = std::string(argv[3]) == "1";
    check(hiddenMip || std::string(argv[3]) == "0", "hidden-mip must be 0 or 1");
    const char *rdocPath = std::getenv("PVRGPU_RENDERDOC_LIB");
    const char *glesPath = std::getenv("RENDERDOC_MESA_GLES_PATH");
    check(rdocPath && glesPath, "Explicit RenderDoc and Mesa GLES libraries required");
    void *rdoc = dlopen(rdocPath, RTLD_NOW | RTLD_LOCAL);
    void *gles = dlopen(glesPath, RTLD_NOW | RTLD_LOCAL);
    check(rdoc && gles, "Cannot open explicit libraries");
    auto getAPI = reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(rdoc, "RENDERDOC_GetAPI"));
    RENDERDOC_API_1_7_0 *api = nullptr;
    check(getAPI && getAPI(eRENDERDOC_API_Version_1_7_0, reinterpret_cast<void **>(&api)) && api,
          "RenderDoc capture API 1.7 unavailable");
    api->SetCaptureOptionU32(eRENDERDOC_Option_APIValidation, 1);
    api->SetCaptureOptionU32(eRENDERDOC_Option_RefAllResources, 1);
    api->SetCaptureOptionU32(eRENDERDOC_Option_SaveAllInitials, 1);
    api->MaskOverlayBits(0, 0);
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major = 0, minor = 0, count = 0;
    check(display != EGL_NO_DISPLAY && eglInitialize(display, &major, &minor), "EGL initialize");
    const EGLint attrs[] = {EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
    EGLConfig config = nullptr;
    check(eglBindAPI(EGL_OPENGL_ES_API) && eglChooseConfig(display,attrs,&config,1,&count) && count == 1, "EGL config");
    const EGLint contextAttrs[] = {EGL_CONTEXT_MAJOR_VERSION_KHR,3,EGL_CONTEXT_MINOR_VERSION_KHR,1,EGL_NONE};
    const EGLint surfaceAttrs[] = {EGL_WIDTH,8,EGL_HEIGHT,8,EGL_NONE};
    EGLContext context = eglCreateContext(display,config,EGL_NO_CONTEXT,contextAttrs);
    EGLSurface surface = eglCreatePbufferSurface(display,config,surfaceAttrs);
    check(context != EGL_NO_CONTEXT && surface != EGL_NO_SURFACE &&
          eglMakeCurrent(display,surface,surface,context), "EGL current");
    const std::string renderer = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
    const std::string version = reinterpret_cast<const char *>(glGetString(GL_VERSION));
    std::cerr << "renderer=" << renderer << " version=" << version << '\n';
    const char *vsSource = "#version 310 es\nlayout(location=0) in highp vec4 position;\n"
      "out highp vec2 uv; void main(){gl_Position=position;uv=position.xy*0.5+0.5;}\n";
    const char *fsSource = "#version 310 es\nprecision highp float;\n"
      "in highp vec2 uv; layout(binding=0) uniform highp sampler2D image;\n"
      "layout(std140,binding=2) uniform Parameters { highp vec4 scale; highp vec4 bias; };\n"
      "layout(location=0) out highp vec4 color;\n"
      "void main(){color=texture(image,uv)*scale+bias;}\n";
    const char *hiddenFsSource = "#version 310 es\nprecision highp float;\n"
      "in highp vec2 uv; layout(binding=0) uniform highp sampler2D image;\n"
      "layout(std140,binding=2) uniform Parameters { highp vec4 scale; highp vec4 bias; };\n"
      "uniform highp float explicitLod; layout(location=0) out highp vec4 color;\n"
      "void main(){color=textureLod(image,uv,explicitLod)*scale+bias;}\n";
    const GLuint vs = shader(GL_VERTEX_SHADER,vsSource), fs = shader(GL_FRAGMENT_SHADER,hiddenMip ? hiddenFsSource : fsSource);
    const GLuint program = glCreateProgram();
    glAttachShader(program,vs); glAttachShader(program,fs); glLinkProgram(program);
    GLint linked = 0; glGetProgramiv(program,GL_LINK_STATUS,&linked);
    if(!linked) { char log[8192] = {}; glGetProgramInfoLog(program,sizeof(log),nullptr,log); throw std::runtime_error(log); }
    glUseProgram(program);
    const GLint lodLocation = hiddenMip ? glGetUniformLocation(program,"explicitLod") : -1;
    check(!hiddenMip || lodLocation >= 0,"explicit LOD uniform missing");
    GLuint textures[5] = {}, fbos[3] = {}, vao = 0, vbo = 0, ubo = 0;
    glGenTextures(depth ? 5 : 4,textures); glGenFramebuffers(3,fbos);
    std::array<uint8_t,256> pixels{};
    for(unsigned image = 0; image < 4; ++image) {
      for(unsigned y = 0; y < 8; ++y) for(unsigned x = 0; x < 8; ++x) {
        const unsigned offset = 4*(8*y+x);
        pixels[offset] = image == 0 ? uint8_t(8+16*x) : 12;
        pixels[offset+1] = image == 0 ? uint8_t(24+20*y) : 20;
        pixels[offset+2] = image == 0 ? uint8_t(16+8*((x+2*y)%8)) : 28;
        pixels[offset+3] = 255;
      }
      glBindTexture(GL_TEXTURE_2D,textures[image]);
      if(hiddenMip && image == 1) {
        // Draw0 produces mip1 while GL_MAX_LEVEL hides it from sampling.
        // Draw1 exposes and consumes it: saving only currently visible mips
        // cannot implement a checkpoint of the physical resource storage.
        glTexStorage2D(GL_TEXTURE_2D,2,GL_RGBA8,16,16);
        std::array<uint8_t,1024> base{};
        for(unsigned i = 0; i < base.size(); ++i) base[i] = pixels[i % 4];
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,16,16,GL_RGBA,GL_UNSIGNED_BYTE,base.data());
        glTexSubImage2D(GL_TEXTURE_2D,1,0,0,8,8,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,0);
      } else {
        glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA8,8,8);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,8,8,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
      }
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,hiddenMip && image == 1 ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    }
    if(depth) {
      glBindTexture(GL_TEXTURE_2D,textures[4]);
      glTexStorage2D(GL_TEXTURE_2D,1,GL_DEPTH_COMPONENT24,8,8);
    }
    for(unsigned i = 0; i < 3; ++i) {
      glBindFramebuffer(GL_FRAMEBUFFER,fbos[i]);
      glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,textures[i+1],hiddenMip && i == 0 ? 1 : 0);
      if(depth) glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,textures[4],0);
      check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,"FBO complete");
    }
    glGenVertexArrays(1,&vao); glBindVertexArray(vao);
    glGenBuffers(1,&vbo); glBindBuffer(GL_ARRAY_BUFFER,vbo);
    std::array<float,24> vertices; positions(vertices,0,depth);
    glBufferData(GL_ARRAY_BUFFER,sizeof(vertices),vertices.data(),GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,nullptr); glEnableVertexAttribArray(0);
    glGenBuffers(1,&ubo); glBindBuffer(GL_UNIFORM_BUFFER,ubo);
    const float initial[] = {1,1,1,1,0,0,0,0};
    glBufferData(GL_UNIFORM_BUFFER,sizeof(initial),initial,GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER,2,ubo);
    glViewport(0,0,8,8); glDisable(GL_BLEND); glDisable(GL_DITHER);
    glDisable(GL_CULL_FACE); glDisable(GL_SCISSOR_TEST);
    if(depth) { glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS); glDepthMask(GL_TRUE); glClearDepthf(1); glClear(GL_DEPTH_BUFFER_BIT); }
    else glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER,fbos[0]);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,textures[0]);
    glFinish(); checkGL("fixture initialization");
    const uint32_t captureIndex = api->GetNumCaptures();
    api->SetCaptureFilePathTemplate((out / "three-draw-dependency").c_str());
    api->StartFrameCapture(nullptr,nullptr);
    check(api->IsFrameCapturing() != 0,"capture did not start");
    // The scene initializes depth within the ORIGINAL capture. This keeps a
    // known initial-depth-capture codec gap separate from the checkpoint test:
    // after draw0, saved nonuniform depth must still govern draw1's write mask.
    if(depth) { glClearDepthf(1); glClear(GL_DEPTH_BUFFER_BIT); }
    std::vector<pvrgpu::rdc::CompletedDrawColor> colors;
    for(unsigned draw = 0; draw < 3; ++draw) {
      glBindFramebuffer(GL_FRAMEBUFFER,fbos[draw]);
      glBindTexture(GL_TEXTURE_2D,textures[draw]);
      if(hiddenMip) {
        if(draw == 1) glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,1);
        glUniform1f(lodLocation,draw == 1 ? 1.f : 0.f);
      }
      positions(vertices,int(draw),depth);
      glBindBuffer(GL_ARRAY_BUFFER,vbo); glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(vertices),vertices.data());
      const float parameters[3][8] = {{1,1,1,1,0,0,0,0},
        {.5f,.5f,.5f,1,32.f/255.f,16.f/255.f,8.f/255.f,0}, {-1,-1,-1,1,1,1,1,0}};
      glBindBuffer(GL_UNIFORM_BUFFER,ubo); glBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(initial),parameters[draw]);
      glDrawArrays(GL_TRIANGLES,0,6);
      glFinish();
      checkGL("ordered draw");
      colors.push_back(pvrgpu::rdc::ReadCompletedDrawColor(gles,out / ("draw"+std::to_string(draw)+".rgba")));
    }
    // OpenCapture may inspect the complete original capture before isolation.
    // Poison A in its real recorded tail so a no-op restore cannot reuse that
    // preview's prefix output and accidentally pass the resumed pixel oracle.
    for(unsigned i = 0; i < pixels.size(); i += 4) {
      pixels[i] = 64; pixels[i+1] = 0; pixels[i+2] = 96; pixels[i+3] = 255;
    }
    glBindTexture(GL_TEXTURE_2D,textures[1]);
    glTexSubImage2D(GL_TEXTURE_2D,hiddenMip ? 1 : 0,0,0,8,8,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
    glFinish(); checkGL("capture complete");
    check(api->EndFrameCapture(nullptr,nullptr) != 0,"capture did not end");
    check(api->GetNumCaptures() == captureIndex+1,"capture count");
    uint32_t length = 0;
    check(api->GetCapture(captureIndex,nullptr,&length,nullptr) && length > 0,"capture path length");
    std::vector<char> capturePath(length+1,0);
    check(api->GetCapture(captureIndex,capturePath.data(),&length,nullptr),"capture path");
    check(fs::is_regular_file(capturePath.data()),"capture artifact missing");

    // Deliberately non-default pack/PBO/read-FBO/texture state independently
    // exercises the engine's readback helper after capture is already closed.
    GLuint packBuffer = 0; glGenBuffers(1,&packBuffer); glBindBuffer(GL_PIXEL_PACK_BUFFER,packBuffer);
    glBufferData(GL_PIXEL_PACK_BUFFER,1024,nullptr,GL_STREAM_READ);
    glPixelStorei(GL_PACK_ALIGNMENT,8); glPixelStorei(GL_PACK_ROW_LENGTH,16);
    glPixelStorei(GL_PACK_SKIP_PIXELS,2); glPixelStorei(GL_PACK_SKIP_ROWS,1);
    for(unsigned i = 0; i < 3; ++i) {
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER,fbos[i]);
      glBindFramebuffer(GL_READ_FRAMEBUFFER,fbos[(i+1)%3]);
      glBindTexture(GL_TEXTURE_2D,textures[0]);
      (void)pvrgpu::rdc::ReadCompletedDrawColor(gles,out / ("state-probe"+std::to_string(i)+".rgba"));
      const GLenum names[] = {GL_DRAW_FRAMEBUFFER_BINDING,GL_READ_FRAMEBUFFER_BINDING,
        GL_PIXEL_PACK_BUFFER_BINDING,GL_TEXTURE_BINDING_2D,GL_PACK_ALIGNMENT,GL_PACK_ROW_LENGTH,GL_PACK_SKIP_PIXELS,GL_PACK_SKIP_ROWS};
      const GLint expected[] = {GLint(fbos[i]),GLint(fbos[(i+1)%3]),GLint(packBuffer),GLint(textures[0]),8,16,2,1};
      for(unsigned j = 0; j < 8; ++j) { GLint value = 0; glGetIntegerv(names[j],&value); check(value == expected[j],"readback state not restored"); }
      checkGL("readback state restored");
    }
    unsigned negativeChecks = 0;
    const auto refused = [&](const fs::path &path) {
      bool rejected = false;
      try { (void)pvrgpu::rdc::ReadCompletedDrawColor(gles,path); }
      catch(const std::exception &) { rejected = true; }
      check(rejected,"unsupported readback accepted"); checkGL("negative readback clean state");
      ++negativeChecks;
    };
    // Exclusive-create never replaces earlier verified pixels.
    refused(out / "draw2.rgba");
    std::string preservedHash, hashError;
    check(pvrgpu::rdc::Sha256File(out / "draw2.rgba",&preservedHash,&hashError) &&
          preservedHash == colors[2].sha256,"existing color output overwritten");
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER,0);
    refused(out / "missing-fbo-must-not-exist.rgba");
    GLuint unsupportedFbo = 0, msaa = 0, narrow = 0;
    glGenFramebuffers(1,&unsupportedFbo); glBindFramebuffer(GL_DRAW_FRAMEBUFFER,unsupportedFbo);
    glGenRenderbuffers(1,&msaa); glBindRenderbuffer(GL_RENDERBUFFER,msaa);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER,4,GL_RGBA8,8,8);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_RENDERBUFFER,msaa);
    checkGL("MSAA negative fixture");
    refused(out / "msaa-must-not-exist.rgba");
    glGenTextures(1,&narrow); glBindTexture(GL_TEXTURE_2D,narrow);
    glTexStorage2D(GL_TEXTURE_2D,1,GL_R8,8,8);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,narrow,0);
    checkGL("non-RGBA8 negative fixture");
    refused(out / "narrow-must-not-exist.rgba");
    for(const char *name : {"missing-fbo-must-not-exist.rgba", "msaa-must-not-exist.rgba", "narrow-must-not-exist.rgba"})
      check(!fs::exists(out / name),"refused readback left a fabricated output");
    std::ofstream result(out / "capture.json");
    result << "{\"schema\":\"pvrgpu.checkpoint-fixture.v1\",\"capture\":"
           << pvrgpu::rdc::drawlist_color_detail::Quote(fs::canonical(capturePath.data()).string())
           << ",\"draws\":3,\"width\":8,\"height\":8,\"depth\":" << (depth ? "true" : "false")
           << ",\"hidden_mip\":" << (hiddenMip ? "true" : "false")
           << ",\"tail_poisoned_prefix_texture\":true"
           << ",\"renderer\":" << pvrgpu::rdc::drawlist_color_detail::Quote(renderer)
           << ",\"version\":" << pvrgpu::rdc::drawlist_color_detail::Quote(version)
           << ",\"readback_negative_checks\":" << negativeChecks
           << ",\"readback_state_restored\":true,\"colors\":[";
    for(size_t i = 0; i < colors.size(); ++i) result << (i ? "," : "") << colors[i].ToJson();
    result << "]}\n"; result.close(); check(bool(result),"capture receipt write");
    std::cout << capturePath.data() << '\n';
    eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
    eglDestroyContext(display,context); eglDestroySurface(display,surface); eglTerminate(display);
    return 0;
  } catch(const std::exception &error) {
    std::cerr << "checkpoint fixture: " << error.what() << '\n'; return 1;
  }
}
