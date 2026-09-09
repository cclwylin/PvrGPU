/* SPDX-License-Identifier: MIT */
/* Real VS instance addressing with a 128-byte UBO-array stride. The shader
 * consumes five attributes and both matrices, leaving register reuse to PCO.
 * Every TF word, guard word and final integer pixel has an independent oracle. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SIDE = 8, INSTANCES = 128, BLOCK_BYTES = INSTANCES * 128,
       GUARD = 8, MAX_RECORDS = 9, TF_WORDS = GUARD * 2 + MAX_RECORDS * 16 };
static unsigned checks, draws;
static void check(int good, const char *message)
{
   ++checks;
   if (!good) { fprintf(stderr,"FAIL: %s\n",message); exit(1); }
}
static void clean(const char *message)
{
   GLenum error = glGetError();
   if (error) fprintf(stderr,"GL_ERROR %s 0x%x\n",message,error);
   check(error == GL_NO_ERROR,message);
}
static uint32_t bits(float value)
{
   uint32_t result; memcpy(&result,&value,sizeof(result)); return result;
}
static GLuint shader(GLenum stage, const char *source)
{
   GLuint s = glCreateShader(stage); glShaderSource(s,1,&source,NULL); glCompileShader(s);
   GLint ok = 0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
   if (!ok) { char log[8192]; glGetShaderInfoLog(s,sizeof(log),NULL,log); fprintf(stderr,"%s\n%s\n",source,log); }
   check(ok,"compile instance-addressing shader"); return s;
}
static GLuint program(void)
{
   const char *vs = "#version 310 es\nprecision highp float;precision highp int;\n"
      "layout(location=0) in vec4 position;layout(location=1) in vec4 normal;"
      "layout(location=2) in vec4 tangent;layout(location=3) in vec4 uv;layout(location=4) in vec4 extra;"
      "struct InstanceData {mat4 current;mat4 previous;};"
      "layout(std140,binding=0) uniform Instances {InstanceData data[128];};uniform int instance_offset;"
      "flat out vec4 observed0;flat out vec4 observed1;flat out vec4 observed2;flat out vec4 observed3;"
      "void main(){int i=gl_InstanceID+instance_offset;gl_Position=position;"
      "observed0=data[i].current*vec4(normal.xyz,1);"
      "observed1=data[i].previous*vec4(tangent.xyz,1);"
      "observed2=vec4(uv.xy,extra.xy);observed3=vec4(float(i),normal.w,tangent.w,uv.z+uv.w);}\n";
   const char *fs = "#version 310 es\nprecision highp float;precision highp int;\n"
      "flat in vec4 observed0;flat in vec4 observed1;flat in vec4 observed2;flat in vec4 observed3;"
      "layout(location=0) out highp uvec4 color;"
      "void main(){color=floatBitsToUint(observed0+observed1+observed2+observed3);}\n";
   GLuint p = glCreateProgram(), v = shader(GL_VERTEX_SHADER,vs), f = shader(GL_FRAGMENT_SHADER,fs);
   glAttachShader(p,v); glAttachShader(p,f); glDeleteShader(v); glDeleteShader(f);
   const char *varyings[] = {"observed0","observed1","observed2","observed3"};
   glTransformFeedbackVaryings(p,4,varyings,GL_INTERLEAVED_ATTRIBS); glLinkProgram(p);
   GLint ok = 0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
   if (!ok) { char log[8192]; glGetProgramInfoLog(p,sizeof(log),NULL,log); fprintf(stderr,"%s\n",log); }
   check(ok,"link instance-addressing shader"); return p;
}
static unsigned word(unsigned buffer, unsigned page, unsigned instance, unsigned matrix, unsigned col, unsigned row)
{
   return 1 + buffer * 4096 + page * 2048 + instance * 8 + matrix * 64 + col * 4 + row;
}
static void expected_record(unsigned buffer, unsigned page, unsigned instance, float result[16])
{
   const int weights[2][4] = {{1,0,-1,1},{0,1,1,1}};
   for (unsigned m = 0; m < 2; ++m) for (unsigned row = 0; row < 4; ++row) {
      int sum = 0;
      for (unsigned col = 0; col < 4; ++col) sum += weights[m][col] * (int)word(buffer,page,instance,m,col,row);
      result[m * 4 + row] = (float)sum;
   }
   result[8]=5; result[9]=6; result[10]=9; result[11]=10;
   result[12]=(float)instance; result[13]=1; result[14]=-1; result[15]=1;
}
int main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display != EGL_NO_DISPLAY && eglInitialize(display,NULL,NULL),"EGL initialize");
   const EGLint attrs[] = {EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
   const EGLint sa[] = {EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};
   const EGLint ca[] = {EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,1,EGL_NONE};
   EGLConfig config = NULL; EGLint count = 0;
   check(eglBindAPI(EGL_OPENGL_ES_API) && eglChooseConfig(display,attrs,&config,1,&count) && count == 1,"EGL config");
   EGLSurface surface = eglCreatePbufferSurface(display,config,sa);
   EGLContext context = eglCreateContext(display,config,EGL_NO_CONTEXT,ca);
   check(surface != EGL_NO_SURFACE && context != EGL_NO_CONTEXT && eglMakeCurrent(display,surface,surface,context),"EGL current");
   fprintf(stderr,"RENDERER: %s\nVERSION: %s\n",glGetString(GL_RENDERER),glGetString(GL_VERSION));
   GLuint p = program(); glUseProgram(p);
   GLint offset = glGetUniformLocation(p,"instance_offset"); check(offset >= 0,"runtime instance offset");
   GLuint block = glGetUniformBlockIndex(p,"Instances"); check(block != GL_INVALID_INDEX,"active UBO");
   GLint block_size = 0; glGetActiveUniformBlockiv(p,block,GL_UNIFORM_BLOCK_DATA_SIZE,&block_size);
   check(block_size == BLOCK_BYTES,"exact std140 128-byte array stride"); glUniformBlockBinding(p,block,0);
   GLint alignment = 0; glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT,&alignment);
   check(alignment > 0 && alignment <= 65536,"bounded UBO range alignment");
   size_t stride = ((BLOCK_BYTES + (size_t)alignment - 1) / (size_t)alignment) * (size_t)alignment;
   GLuint ubo[2], tex, fbo, vao, buffers[3], query;
   glGenBuffers(2,ubo);
   for (unsigned b = 0; b < 2; ++b) {
      unsigned char *storage = malloc(stride * 2); check(storage != NULL,"UBO allocation"); memset(storage,0xa7,stride * 2);
      for (unsigned page = 0; page < 2; ++page) for (unsigned i = 0; i < INSTANCES; ++i)
         for (unsigned m = 0; m < 2; ++m) for (unsigned c = 0; c < 4; ++c) for (unsigned r = 0; r < 4; ++r) {
            float value = (float)word(b,page,i,m,c,r);
            memcpy(storage + page * stride + (i * 32 + m * 16 + c * 4 + r) * 4,&value,4);
         }
      glBindBuffer(GL_UNIFORM_BUFFER,ubo[b]); glBufferData(GL_UNIFORM_BUFFER,(GLsizeiptr)(stride * 2),storage,GL_STATIC_DRAW); free(storage);
   }
   glGenTextures(1,&tex); glBindTexture(GL_TEXTURE_2D,tex); glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA32UI,SIDE,SIDE);
   glGenFramebuffers(1,&fbo); glBindFramebuffer(GL_FRAMEBUFFER,fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,0);
   GLenum attachment = GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&attachment); glReadBuffer(attachment);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,"integer output FBO");
   struct Vertex {float position[3]; int8_t normal[4], tangent[4]; float uv[2], extra[2];};
   _Static_assert(sizeof(struct Vertex) == 36,"exact nine-dword physical vertex");
   const struct Vertex vertices[3] = {
      {{-1,-1,0}, {127,0,-127,127}, {0,127,127,-128}, {5,6}, {9,10}},
      {{ 3,-1,0}, {127,0,-127,127}, {0,127,127,-128}, {5,6}, {9,10}},
      {{-1, 3,0}, {127,0,-127,127}, {0,127,127,-128}, {5,6}, {9,10}},
   };
   const uint16_t indices[] = {0,1,2};
   glGenVertexArrays(1,&vao); glBindVertexArray(vao); glGenBuffers(3,buffers);
   glBindBuffer(GL_ARRAY_BUFFER,buffers[0]); glBufferData(GL_ARRAY_BUFFER,sizeof(vertices),vertices,GL_STATIC_DRAW);
   const unsigned offsets[] = {0,12,16,20,28}, sizes[] = {3,4,4,2,2};
   for (unsigned a = 0; a < 5; ++a) {
      unsigned snorm = a == 1 || a == 2;
      glVertexAttribPointer(a,(GLint)sizes[a],snorm ? GL_BYTE : GL_FLOAT,snorm ? GL_TRUE : GL_FALSE,sizeof(vertices[0]),(void *)(uintptr_t)offsets[a]);
      glEnableVertexAttribArray(a);
   }
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,buffers[1]); glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(indices),indices,GL_STATIC_DRAW);
   glGenQueries(1,&query);
   glViewport(0,0,SIDE,SIDE); glDisable(GL_DITHER); glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST); glDisable(GL_SCISSOR_TEST); glDisable(GL_CULL_FACE);
   clean("setup");
   for (unsigned b = 0; b < 2; ++b) for (unsigned page = 0; page < 2; ++page)
      for (unsigned edge = 0; edge < 2; ++edge) for (unsigned instances = 1; instances <= 3; instances += 2) {
         unsigned first = edge ? 125 : 0; glUniform1i(offset,(GLint)first);
         glBindBufferRange(GL_UNIFORM_BUFFER,0,ubo[b],(GLintptr)(page * stride),BLOCK_BYTES);
         uint32_t initial[TF_WORDS];
         for (unsigned i = 0; i < TF_WORDS; ++i) initial[i] = UINT32_C(0xdead0000) ^ (i * UINT32_C(0x1021));
         glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER,buffers[2]); glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER,sizeof(initial),initial,GL_DYNAMIC_READ);
         glBindBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER,0,buffers[2],GUARD * 4,MAX_RECORDS * 16 * 4);
         const GLuint zero[] = {0,0,0,0}; glClearBufferuiv(GL_COLOR,0,zero);
         glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN,query); glBeginTransformFeedback(GL_TRIANGLES);
         glDrawElementsInstanced(GL_TRIANGLES,3,GL_UNSIGNED_SHORT,NULL,(GLsizei)instances);
         glEndTransformFeedback(); glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN); glFinish(); clean("instance-addressing draw");
         GLuint written = UINT32_MAX; glGetQueryObjectuiv(query,GL_QUERY_RESULT,&written); check(written == instances,"one TF triangle per instance");
         const uint32_t *actual = glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER,0,sizeof(initial),GL_MAP_READ_BIT); check(actual != NULL,"feedback map");
         for (unsigned i = 0; i < TF_WORDS; ++i) {
            uint32_t expected = initial[i];
            if (i >= GUARD && i < GUARD + instances * 3 * 16) {
               unsigned index = i - GUARD; float record[16]; expected_record(b,page,first + index / 48,record); expected = bits(record[index % 16]);
            }
            if (actual[i] != expected) fprintf(stderr,"TF draw%u word%u actual%08x expected%08x\n",draws,i,actual[i],expected);
            check(actual[i] == expected,"all TF words, unwritten tail and guards match independent oracle");
         }
         check(glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER),"feedback unmap");
         float last[16]; uint32_t expected[4]; expected_record(b,page,first + instances - 1,last);
         for (unsigned c = 0; c < 4; ++c) expected[c] = bits(last[c] + last[4+c] + last[8+c] + last[12+c]);
         uint32_t pixels[SIDE * SIDE * 4]; glReadPixels(0,0,SIDE,SIDE,GL_RGBA_INTEGER,GL_UNSIGNED_INT,pixels); clean("integer readback");
         for (unsigned i = 0; i < SIDE * SIDE * 4; ++i) check(pixels[i] == expected[i%4],"all final pixels match last instance independent oracle");
         printf("DRAW %u buffer=%u page=%u first=%u instances=%u rgba=%08x,%08x,%08x,%08x PASS\n",
            draws,b,page,first,instances,expected[0],expected[1],expected[2],expected[3]); fflush(stdout); ++draws;
      }
   glDeleteQueries(1,&query); glDeleteBuffers(2,ubo); glDeleteBuffers(3,buffers); glDeleteVertexArrays(1,&vao);
   glDeleteFramebuffers(1,&fbo); glDeleteTextures(1,&tex); glDeleteProgram(p); clean("cleanup");
   printf("PASS draws=%u checks=%u\n",draws,checks);
   eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT); eglDestroyContext(display,context); eglDestroySurface(display,surface); eglTerminate(display);
   return 0;
}
