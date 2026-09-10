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

static std::array<unsigned,3> rgb9Codes(unsigned seed, unsigned x, unsigned y)
{
  if((x+3*y+seed)%11 == 0) return {{0,0,0}};
  constexpr unsigned values[] = {0,1,2,4};
  return {{values[(x+y+seed)%4], values[(x+2*y+seed+1)%4], values[(2*x+y+seed+2)%4]}};
}

static std::vector<uint32_t> rgb9Pattern(unsigned seed, unsigned level)
{
  const unsigned extent = 16 >> level;
  std::vector<uint32_t> words(extent*extent);
  for(unsigned y = 0; y < extent; ++y) for(unsigned x = 0; x < extent; ++x) {
    const auto codes = rgb9Codes(seed,x,y);
    const bool zero = codes[0] == 0 && codes[1] == 0 && codes[2] == 0;
    const unsigned shift = (x+y+seed)%4, exponent = zero ? 31 : 15+shift;
    // Value is code/8. Extra exponent headroom is intentional: a float
    // unpack/repack can preserve the sampled value while changing these bits.
    words[y*extent+x] = (zero ? 0 : ((codes[0]*64)>>shift) |
      (((codes[1]*64)>>shift)<<9) | (((codes[2]*64)>>shift)<<18)) | (exponent<<27);
  }
  return words;
}

// Test observation only, using the explicit GLES library rather than recording
// additional replay calls. D24's GL_UNSIGNED_INT readback is a normalized U32
// representation, not the driver's native 24-bit storage layout.
static pvrgpu::rdc::CompletedDrawColor readDepthCodes(void *library, const fs::path &path,
                                                    unsigned width, unsigned height)
{
  using namespace pvrgpu::rdc;
  drawlist_color_detail::GL gl(library);
  drawlist_color_detail::Restore restore(gl);
  gl.Check("depth observation entry");
  gl.bindFramebuffer(GL_READ_FRAMEBUFFER,unsigned(gl.Integer(GL_DRAW_FRAMEBUFFER_BINDING)));
  gl.bindBuffer(GL_PIXEL_PACK_BUFFER,0);
  for(unsigned parameter : {GL_PACK_ALIGNMENT,GL_PACK_ROW_LENGTH,GL_PACK_SKIP_PIXELS,GL_PACK_SKIP_ROWS})
    gl.pack(parameter,parameter == GL_PACK_ALIGNMENT ? 1 : 0);
  if(restore.reverseAvailable) gl.pack(0x93A4,0);
  std::vector<uint32_t> words(width*height);
  gl.readPixels(0,0,int(width),int(height),GL_DEPTH_COMPONENT,GL_UNSIGNED_INT,words.data());
  gl.Check("D24 raw code observation");
  restore.Apply(); gl.Check("depth observation restoration");
  std::vector<uint8_t> bytes(words.size()*4);
  for(size_t i = 0; i < words.size(); ++i)
    for(unsigned byte = 0; byte < 4; ++byte) bytes[4*i+byte] = uint8_t(words[i] >> (8*byte));
  drawlist_color_detail::ExclusiveWrite(path,bytes);
  CompletedDrawColor result;
  result.path = path.string(); result.width = width; result.height = height;
  result.size_bytes = bytes.size(); result.format = "D24_GL_UNSIGNED_INT_LE";
  result.source = "completed-producer-depth";
  std::string error;
  check(Sha256File(path,&result.sha256,&error),"depth observation hash: "+error);
  return result;
}

int main(int argc, char **argv)
{
  try {
    check(argc >= 4 && argc <= 8, "Usage: checkpoint-capture EXISTING_OUTPUT_DIRECTORY depth(0|1) hidden-mip(0|1) [depth-renderbuffer(0|1) [depth-low-codes(0|1) [buffer-target-alias(0|1) [rgb9e5(0|1)]]]]");
    const fs::path out = fs::canonical(argv[1]);
    const bool depth = std::string(argv[2]) == "1";
    check(depth || std::string(argv[2]) == "0", "depth must be 0 or 1");
    const bool hiddenMip = std::string(argv[3]) == "1";
    check(hiddenMip || std::string(argv[3]) == "0", "hidden-mip must be 0 or 1");
    const bool depthRenderbuffer = argc >= 5 && std::string(argv[4]) == "1";
    check(argc == 4 || depthRenderbuffer || std::string(argv[4]) == "0", "depth-renderbuffer must be 0 or 1");
    const bool depthLowCodes = argc >= 6 && std::string(argv[5]) == "1";
    check(argc < 6 || depthLowCodes || std::string(argv[5]) == "0", "depth-low-codes must be 0 or 1");
    check(!depthRenderbuffer || (!depth && !hiddenMip), "depth-renderbuffer is a separate fixture mode");
    check(!depthLowCodes || (!depth && !depthRenderbuffer && !hiddenMip), "depth-low-codes is a separate fixture mode");
    const bool bufferTargetAlias = argc >= 7 && std::string(argv[6]) == "1";
    check(argc < 7 || bufferTargetAlias || std::string(argv[6]) == "0", "buffer-target-alias must be 0 or 1");
    check(!bufferTargetAlias || (!depth && !depthRenderbuffer && !depthLowCodes && !hiddenMip),
          "buffer-target-alias is a separate fixture mode");
    const bool rgb9e5 = argc == 8 && std::string(argv[7]) == "1";
    check(argc < 8 || rgb9e5 || std::string(argv[7]) == "0", "rgb9e5 must be 0 or 1");
    check(!rgb9e5 || (!depth && !depthRenderbuffer && !depthLowCodes && !hiddenMip && !bufferTargetAlias),
          "rgb9e5 is a separate fixture mode");
    const bool aliasedDepth = depthRenderbuffer || depthLowCodes, depthTexture = depth || depthLowCodes;
    const bool hasDepth = depthTexture || depthRenderbuffer;
    const unsigned width = aliasedDepth ? 7 : 8, height = aliasedDepth ? 5 : 8;
    const unsigned fboCount = aliasedDepth ? 2 : 3;
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
    const EGLint surfaceAttrs[] = {EGL_WIDTH,EGLint(width),EGL_HEIGHT,EGLint(height),EGL_NONE};
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
    const char *renderbufferFsSource = "#version 310 es\nprecision highp float;\n"
      "in highp vec2 uv; layout(binding=0) uniform highp sampler2D image;\n"
      "layout(std140,binding=2) uniform Parameters { highp vec4 scale; highp vec4 bias; };\n"
      "uniform highp uint depthPass; layout(location=0) out highp vec4 color;\n"
      "void main(){highp uint cell=(3u*uint(gl_FragCoord.x)+5u*uint(gl_FragCoord.y))%4u;\n"
      "gl_FragDepth=depthPass==0u ? float(2u*cell+1u)*0.125 :\n"
      "depthPass==1u ? 0.5 : cell==3u ? 0.75 : 0.25;\n"
      "color=texture(image,uv)*scale+bias;}\n";
    const char *lowDepthFsSource = "#version 310 es\nprecision highp float;\n"
      "in highp vec2 uv; layout(binding=0) uniform highp sampler2D image;\n"
      "layout(std140,binding=2) uniform Parameters { highp vec4 scale; highp vec4 bias; };\n"
      "uniform highp uint depthPass; layout(location=0) out highp vec4 color;\n"
      "void main(){highp uint cell=(3u*uint(gl_FragCoord.x)+5u*uint(gl_FragCoord.y))%4u;\n"
      "highp uint code=depthPass==0u ? (cell==0u ? 1u : cell==1u ? 159u : cell==2u ? 256u : 65535u) :\n"
      "depthPass==1u ? 160u : cell==3u ? 200u : 100u;\n"
      "gl_FragDepth=float(code)/16777215.0; color=texture(image,uv)*scale+bias;}\n";
    const char *rgb9FsSource = "#version 310 es\nprecision highp float;\n"
      "in highp vec2 uv; layout(binding=0) uniform highp sampler2D image;\n"
      "layout(binding=1) uniform highp sampler2D packedImage; uniform highp uint rgbPass;\n"
      "layout(location=0) out highp vec4 color;\n"
      "void main(){highp vec3 p=textureLod(packedImage,uv,1.0).rgb;\n"
      "color=vec4(rgbPass==0u ? p : texture(image,uv).rgb*0.5+p*0.25,1.0);}\n";
    const GLuint vs = shader(GL_VERTEX_SHADER,vsSource), fs = shader(GL_FRAGMENT_SHADER,
        rgb9e5 ? rgb9FsSource : depthLowCodes ? lowDepthFsSource : depthRenderbuffer ? renderbufferFsSource : hiddenMip ? hiddenFsSource : fsSource);
    const GLuint program = glCreateProgram();
    glAttachShader(program,vs); glAttachShader(program,fs); glLinkProgram(program);
    GLint linked = 0; glGetProgramiv(program,GL_LINK_STATUS,&linked);
    if(!linked) { char log[8192] = {}; glGetProgramInfoLog(program,sizeof(log),nullptr,log); throw std::runtime_error(log); }
    glUseProgram(program);
    const GLint lodLocation = hiddenMip ? glGetUniformLocation(program,"explicitLod") : -1;
    check(!hiddenMip || lodLocation >= 0,"explicit LOD uniform missing");
    const GLint depthPassLocation = aliasedDepth ? glGetUniformLocation(program,"depthPass") : -1;
    check(!aliasedDepth || depthPassLocation >= 0,"depth pass uniform missing");
    const GLint rgbPassLocation = rgb9e5 ? glGetUniformLocation(program,"rgbPass") : -1;
    check(!rgb9e5 || rgbPassLocation >= 0,"RGB9E5 pass uniform missing");
    GLuint textures[5] = {}, fbos[3] = {}, vao = 0, vbo = 0, ubo = 0, depthRb = 0;
    GLuint packedTextures[2] = {};
    std::array<std::array<std::vector<uint32_t>,2>,2> packedWords;
    GLint rbWidth = 0, rbHeight = 0, rbFormat = 0, rbSamples = -1;
    glGenTextures(depthTexture ? 5 : 4,textures); glGenFramebuffers(fboCount,fbos);
    std::vector<uint8_t> pixels(width*height*4);
    for(unsigned image = 0; image < 4; ++image) {
      for(unsigned y = 0; y < height; ++y) for(unsigned x = 0; x < width; ++x) {
        const unsigned offset = 4*(width*y+x);
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
        glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA8,width,height);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
      }
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,hiddenMip && image == 1 ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    }
    if(depthTexture) {
      glBindTexture(GL_TEXTURE_2D,textures[4]);
      glTexStorage2D(GL_TEXTURE_2D,1,GL_DEPTH_COMPONENT24,width,height);
    }
    if(depthRenderbuffer) {
      glGenRenderbuffers(1,&depthRb); glBindRenderbuffer(GL_RENDERBUFFER,depthRb);
      glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH_COMPONENT16,width,height);
      glGetRenderbufferParameteriv(GL_RENDERBUFFER,GL_RENDERBUFFER_WIDTH,&rbWidth);
      glGetRenderbufferParameteriv(GL_RENDERBUFFER,GL_RENDERBUFFER_HEIGHT,&rbHeight);
      glGetRenderbufferParameteriv(GL_RENDERBUFFER,GL_RENDERBUFFER_INTERNAL_FORMAT,&rbFormat);
      glGetRenderbufferParameteriv(GL_RENDERBUFFER,GL_RENDERBUFFER_SAMPLES,&rbSamples);
      checkGL("D16 renderbuffer storage");
      check(rbWidth == GLint(width) && rbHeight == GLint(height) &&
            rbFormat == GL_DEPTH_COMPONENT16 && rbSamples == 0,"exact single-sample D16 storage");
    }
    if(rgb9e5) {
      glGenTextures(2,packedTextures); glActiveTexture(GL_TEXTURE1);
      for(unsigned image = 0; image < 2; ++image) {
        glBindTexture(GL_TEXTURE_2D,packedTextures[image]);
        glTexStorage2D(GL_TEXTURE_2D,2,GL_RGB9_E5,16,16);
        for(unsigned level = 0; level < 2; ++level) {
          const unsigned extent = 16 >> level;
          const std::vector<uint32_t> zero(extent*extent,0);
          glTexSubImage2D(GL_TEXTURE_2D,GLint(level),0,0,extent,extent,GL_RGB,GL_UNSIGNED_INT_5_9_9_9_REV,zero.data());
          packedWords[image][level] = rgb9Pattern(image,level);
          std::vector<uint8_t> bytes(packedWords[image][level].size()*4);
          for(size_t i = 0; i < packedWords[image][level].size(); ++i)
            for(unsigned b = 0; b < 4; ++b) bytes[4*i+b] = uint8_t(packedWords[image][level][i] >> (8*b));
          // These are input-upload artifacts, not claimed GPU observations.
          pvrgpu::rdc::drawlist_color_detail::ExclusiveWrite(out / ("rgb9e5-input"+
            std::to_string(image)+"-mip"+std::to_string(level)+".u32"),bytes);
          GLint w = 0, h = 0, format = 0;
          glGetTexLevelParameteriv(GL_TEXTURE_2D,GLint(level),GL_TEXTURE_WIDTH,&w);
          glGetTexLevelParameteriv(GL_TEXTURE_2D,GLint(level),GL_TEXTURE_HEIGHT,&h);
          glGetTexLevelParameteriv(GL_TEXTURE_2D,GLint(level),GL_TEXTURE_INTERNAL_FORMAT,&format);
          check(w == GLint(extent) && h == GLint(extent) && format == GL_RGB9_E5,"actual RGB9E5 mip allocation");
        }
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,1);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST_MIPMAP_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
      }
      glActiveTexture(GL_TEXTURE0); checkGL("RGB9E5 sampler-only allocations");
    }
    for(unsigned i = 0; i < fboCount; ++i) {
      glBindFramebuffer(GL_FRAMEBUFFER,fbos[i]);
      glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,textures[i+1],hiddenMip && i == 0 ? 1 : 0);
      if(depthTexture) glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,textures[4],0);
      if(depthLowCodes) {
        GLint kind = 0, name = 0, bits = 0;
        glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE,&kind);
        glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,&name);
        glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE,&bits);
        checkGL("shared low-code D24 texture identity");
        check(kind == GL_TEXTURE && GLuint(name) == textures[4] && bits == 24,"FBO aliases must share the same real D24 texture");
      }
      if(depthRenderbuffer) {
        glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_RENDERBUFFER,depthRb);
        GLint kind = 0, name = 0;
        glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE,&kind);
        glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,&name);
        checkGL("shared renderbuffer attachment identity");
        check(kind == GL_RENDERBUFFER && GLuint(name) == depthRb,"FBO aliases must share the same real renderbuffer");
      }
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
    std::vector<GLenum> observedAliasTargets;
    const auto bindBufferAlias = [&](GLenum target) {
      // Same allocated UBO, still consumed via indexed UNIFORM binding2.
      // Generic bindings alone do not dispatch compute or execute indirect
      // commands. There is no reallocation or reinterpretation of its bytes.
      glBindBuffer(target,ubo);
      const GLenum selector = target == GL_DRAW_INDIRECT_BUFFER ? GL_DRAW_INDIRECT_BUFFER_BINDING :
                              target == GL_SHADER_STORAGE_BUFFER ? GL_SHADER_STORAGE_BUFFER_BINDING :
                              GL_COPY_READ_BUFFER_BINDING;
      GLint name = 0, size = 0, indexedUniform = 0;
      glGetIntegerv(selector,&name);
      glGetBufferParameteriv(target,GL_BUFFER_SIZE,&size);
      glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING,2,&indexedUniform);
      checkGL("same-object buffer target alias");
      check(GLuint(name) == ubo && GLuint(indexedUniform) == ubo && size == GLint(sizeof(initial)),
            "buffer target change must preserve physical UBO and indexed binding");
      observedAliasTargets.push_back(target);
    };
    glViewport(0,0,width,height); glDisable(GL_BLEND); glDisable(GL_DITHER);
    glDisable(GL_CULL_FACE); glDisable(GL_SCISSOR_TEST);
    if(hasDepth) { glEnable(GL_DEPTH_TEST); glDepthFunc(GL_LESS); glDepthMask(GL_TRUE); glClearDepthf(1); glClear(GL_DEPTH_BUFFER_BIT); }
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
    if(hasDepth) { glClearDepthf(1); glClear(GL_DEPTH_BUFFER_BIT); }
    std::vector<pvrgpu::rdc::CompletedDrawColor> colors, depthCodes;
    for(unsigned draw = 0; draw < 3; ++draw) {
      glBindFramebuffer(GL_FRAMEBUFFER,fbos[aliasedDepth ? draw % 2 : draw]);
      if(aliasedDepth) {
        if(draw == 2) glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,textures[3],0);
        glUniform1ui(depthPassLocation,draw);
        check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,"aliased FBO remains complete");
        checkGL("renderbuffer draw state");
      }
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
      if(bufferTargetAlias) bindBufferAlias(draw == 1 ? GL_SHADER_STORAGE_BUFFER : GL_DRAW_INDIRECT_BUFFER);
      if(rgb9e5) {
        glActiveTexture(GL_TEXTURE1);
        if(draw < 2) {
          // Original captured uploads belong to this draw's prefix. Draw1
          // consumes saved RGB0, while its new RGB1 must survive save1 for2.
          glBindTexture(GL_TEXTURE_2D,packedTextures[draw]);
          for(unsigned level = 0; level < 2; ++level) {
            const unsigned extent = 16 >> level;
            glTexSubImage2D(GL_TEXTURE_2D,GLint(level),0,0,extent,extent,GL_RGB,
                           GL_UNSIGNED_INT_5_9_9_9_REV,packedWords[draw][level].data());
          }
        }
        glBindTexture(GL_TEXTURE_2D,packedTextures[draw == 2 ? 1 : 0]);
        glActiveTexture(GL_TEXTURE0); glUniform1ui(rgbPassLocation,draw);
        checkGL("original RGB9E5 upload and sample bindings");
      }
      glDrawArrays(GL_TRIANGLES,0,6);
      glFinish();
      checkGL("ordered draw");
      colors.push_back(pvrgpu::rdc::ReadCompletedDrawColor(gles,out / ("draw"+std::to_string(draw)+".rgba")));
      check(colors.back().width == width && colors.back().height == height,"actual draw readback extent");
      if(depthLowCodes) depthCodes.push_back(readDepthCodes(gles,out / ("draw"+std::to_string(draw)+".depth-u32"),width,height));
    }
    // OpenCapture may inspect the complete original capture before isolation.
    // Poison A in its real recorded tail so a no-op restore cannot reuse that
    // preview's prefix output and accidentally pass the resumed pixel oracle.
    for(unsigned i = 0; i < pixels.size(); i += 4) {
      pixels[i] = 64; pixels[i+1] = 0; pixels[i+2] = 96; pixels[i+3] = 255;
    }
    glBindTexture(GL_TEXTURE_2D,textures[1]);
    glTexSubImage2D(GL_TEXTURE_2D,hiddenMip ? 1 : 0,0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
    // Also poison the shared RB in the ORIGINAL tail. OpenCapture preview
    // depth cannot accidentally substitute for either saved checkpoint.
    if(aliasedDepth) { glClearDepthf(0); glClear(GL_DEPTH_BUFFER_BIT); }
    // The complete capture's final metadata deliberately differs from save0
    // and save1; a fresh-controller preview cannot supply either saved target.
    if(bufferTargetAlias) bindBufferAlias(GL_COPY_READ_BUFFER);
    if(rgb9e5) {
      glActiveTexture(GL_TEXTURE1);
      for(GLuint texture : packedTextures) {
        glBindTexture(GL_TEXTURE_2D,texture);
        for(unsigned level = 0; level < 2; ++level) {
          const unsigned extent = 16 >> level;
          const std::vector<uint32_t> zero(extent*extent,0);
          glTexSubImage2D(GL_TEXTURE_2D,GLint(level),0,0,extent,extent,GL_RGB,GL_UNSIGNED_INT_5_9_9_9_REV,zero.data());
        }
      }
      glActiveTexture(GL_TEXTURE0);
    }
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
    for(unsigned i = 0; i < fboCount; ++i) {
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER,fbos[i]);
      glBindFramebuffer(GL_READ_FRAMEBUFFER,fbos[(i+1)%fboCount]);
      glBindTexture(GL_TEXTURE_2D,textures[0]);
      (void)pvrgpu::rdc::ReadCompletedDrawColor(gles,out / ("state-probe"+std::to_string(i)+".rgba"));
      const GLenum names[] = {GL_DRAW_FRAMEBUFFER_BINDING,GL_READ_FRAMEBUFFER_BINDING,
        GL_PIXEL_PACK_BUFFER_BINDING,GL_TEXTURE_BINDING_2D,GL_PACK_ALIGNMENT,GL_PACK_ROW_LENGTH,GL_PACK_SKIP_PIXELS,GL_PACK_SKIP_ROWS};
      const GLint expected[] = {GLint(fbos[i]),GLint(fbos[(i+1)%fboCount]),GLint(packBuffer),GLint(textures[0]),8,16,2,1};
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
           << ",\"draws\":3,\"width\":" << width << ",\"height\":" << height << ",\"depth\":" << (hasDepth ? "true" : "false")
           << ",\"hidden_mip\":" << (hiddenMip ? "true" : "false")
           << ",\"depth_renderbuffer\":" << (depthRenderbuffer ? "true" : "false")
           << ",\"depth_low_codes\":" << (depthLowCodes ? "true" : "false")
           << ",\"buffer_target_alias\":" << (bufferTargetAlias ? "true" : "false")
           << ",\"buffer_target_alias_name\":" << (bufferTargetAlias ? ubo : 0)
           << ",\"buffer_target_alias_size\":" << (bufferTargetAlias ? sizeof(initial) : 0)
           << ",\"buffer_target_alias_verified\":" << (bufferTargetAlias && observedAliasTargets.size() == 4 ? "true" : "false")
           << ",\"rgb9e5\":" << (rgb9e5 ? "true" : "false")
           << ",\"rgb9e5_texture_names\":[" << packedTextures[0] << "," << packedTextures[1] << "]"
           << ",\"rgb9e5_levels\":" << (rgb9e5 ? 2 : 0)
           << ",\"rgb9e5_sampled_mip\":" << (rgb9e5 ? 1 : 0)
           << ",\"rgb9e5_tail_poisoned\":" << (rgb9e5 ? "true" : "false")
           << ",\"depth_texture_name\":" << (depthTexture ? textures[4] : 0)
           << ",\"depth_renderbuffer_name\":" << depthRb
           << ",\"depth_renderbuffer_internal_format\":" << rbFormat
           << ",\"depth_renderbuffer_samples\":" << rbSamples
           << ",\"depth_alias_fbos\":" << (aliasedDepth ? fboCount : 0)
           << ",\"depth_alias_verified\":" << (aliasedDepth ? "true" : "false")
           << ",\"tail_poisoned_depth\":" << (aliasedDepth ? "true" : "false")
           << ",\"tail_poisoned_prefix_texture\":true"
           << ",\"renderer\":" << pvrgpu::rdc::drawlist_color_detail::Quote(renderer)
           << ",\"version\":" << pvrgpu::rdc::drawlist_color_detail::Quote(version)
           << ",\"readback_negative_checks\":" << negativeChecks
           << ",\"readback_state_restored\":true,\"colors\":[";
    for(size_t i = 0; i < colors.size(); ++i) result << (i ? "," : "") << colors[i].ToJson();
    result << "],\"depth_codes\":[";
    for(size_t i = 0; i < depthCodes.size(); ++i) result << (i ? "," : "") << depthCodes[i].ToJson();
    result << "],\"buffer_target_alias_targets\":[";
    for(size_t i = 0; i < observedAliasTargets.size(); ++i) result << (i ? "," : "") << observedAliasTargets[i];
    result << "]}\n"; result.close(); check(bool(result),"capture receipt write");
    std::cout << capturePath.data() << '\n';
    eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
    eglDestroyContext(display,context); eglDestroySurface(display,surface); eglTerminate(display);
    return 0;
  } catch(const std::exception &error) {
    std::cerr << "checkpoint fixture: " << error.what() << '\n'; return 1;
  }
}
