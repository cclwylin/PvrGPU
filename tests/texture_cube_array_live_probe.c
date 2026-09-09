/* SPDX-License-Identifier: MIT */
/* Manual EGL/GLES live CubeArray regression. No capture pixels are inputs.
 * Two cubes x six faces, three explicit RGBA8 mips, asymmetric binary texels.
 * 192 nearest cases have a source-derived exact CPU oracle. Another 156
 * linear/seam/corner/spatial-derivative cases expose every actual binary32
 * word for differential analysis; their PASS means GL/guard/inventory success,
 * not an independently proved seamless interpolation oracle or LP tolerance.
 * Compile: cc -std=c11 -O2 -Wall -Wextra -Werror ... -lEGL -lGLESv2.
 */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SIDE=2, PIXELS=4, GUARD=16, CUBES=2, FACES=6, LEVELS=3 };
static unsigned checks, draws, oracle_draws;
static void check(int ok,const char *what)
{
   ++checks;
   if(!ok){fprintf(stderr,"FAIL: %s\n",what);exit(1);}
}
static void check_gl(const char *what)
{
   GLenum e=glGetError();
   if(e)fprintf(stderr,"GL_ERROR: %s 0x%x\n",what,e);
   check(e==GL_NO_ERROR,what);
}
static uint32_t bits(float v){uint32_t u;memcpy(&u,&v,4);return u;}
static GLuint compile(GLenum stage,const char *source)
{
   GLuint s=glCreateShader(stage);glShaderSource(s,1,&source,NULL);glCompileShader(s);
   GLint ok=0;glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
   if(!ok){char log[8192];glGetShaderInfoLog(s,sizeof(log),NULL,log);fprintf(stderr,"SHADER: %s\n%s\n",log,source);}
   check(ok,"compile CubeArray shader");return s;
}
typedef struct {
   GLuint program;
   GLint coordinate, lod, dx, dy;
} Program;
static Program make_program(int explicit_lod)
{
   const char *vs="#version 310 es\nlayout(location=0) in highp vec4 position;\n"
      "out highp vec2 pixel_position;\n"
      "void main(){gl_Position=position;pixel_position=position.xy;}\n";
   char fs[2048];
   snprintf(fs,sizeof(fs),"#version 310 es\n#extension GL_EXT_texture_cube_map_array : require\n"
      "precision highp float; precision highp int;\n"
      "uniform highp samplerCubeArray source_texture; uniform highp vec4 coordinate;\n"
      "uniform highp float selected_lod; uniform highp vec3 gradient_x,gradient_y;\n"
      "in highp vec2 pixel_position; layout(location=0) out highp uvec4 color;\n"
      "void main(){vec4 c=vec4(coordinate.xyz+gradient_x*pixel_position.x+gradient_y*pixel_position.y,coordinate.w);"
      "color=floatBitsToUint(%s);}\n",explicit_lod?"textureLod(source_texture,c,selected_lod)":"texture(source_texture,c)");
   GLuint v=compile(GL_VERTEX_SHADER,vs),f=compile(GL_FRAGMENT_SHADER,fs);
   Program p={0};p.program=glCreateProgram();glAttachShader(p.program,v);glAttachShader(p.program,f);
   glDeleteShader(v);glDeleteShader(f);glLinkProgram(p.program);
   GLint ok=0;glGetProgramiv(p.program,GL_LINK_STATUS,&ok);
   if(!ok){char log[8192];glGetProgramInfoLog(p.program,sizeof(log),NULL,log);fprintf(stderr,"LINK: %s\n",log);}
   check(ok,"link CubeArray program");glUseProgram(p.program);
   GLint sampler=glGetUniformLocation(p.program,"source_texture");
   p.coordinate=glGetUniformLocation(p.program,"coordinate");p.lod=glGetUniformLocation(p.program,"selected_lod");
   p.dx=glGetUniformLocation(p.program,"gradient_x");p.dy=glGetUniformLocation(p.program,"gradient_y");
   check(sampler>=0&&p.coordinate>=0&&p.dx>=0&&p.dy>=0,"active sample/coordinate/gradient uniforms");
   check(explicit_lod?p.lod>=0:p.lod==-1,"real dynamic LOD only in explicit variant");
   glUniform1i(sampler,0);return p;
}
static unsigned texel_byte(unsigned cube,unsigned face,unsigned mip,unsigned x,unsigned y,unsigned channel)
{
   unsigned bit=0;
   if(channel==0)bit=(face&1u)^cube^((y>>1)&1u);
   if(channel==1)bit=((face>>1)&1u)^(x&1u)^(mip&1u);
   if(channel==2)bit=((face>>2)&1u)^(y&1u)^((x>>1)&1u);
   if(channel==3)bit=(x/2+y/2+cube+mip)&1u;
   return bit?255u:0u;
}
/* GL cube face basis; sc/tc are signed face coordinates. */
static void direction(unsigned face,float sc,float tc,float d[3])
{
   switch(face){
   case 0:d[0]=1;d[1]=-tc;d[2]=-sc;break;
   case 1:d[0]=-1;d[1]=-tc;d[2]=sc;break;
   case 2:d[0]=sc;d[1]=1;d[2]=tc;break;
   case 3:d[0]=sc;d[1]=-1;d[2]=-tc;break;
   case 4:d[0]=sc;d[1]=-tc;d[2]=1;break;
   default:d[0]=-sc;d[1]=-tc;d[2]=-1;break;
   }
}
static void nearest_oracle(const float c[4],unsigned mip,uint32_t expected[4])
{
   float ax=fabsf(c[0]),ay=fabsf(c[1]),az=fabsf(c[2]),ma,sc,tc;unsigned face;
   if(ax>ay&&ax>az){ma=ax;face=c[0]>0?0:1;sc=c[0]>0?-c[2]:c[2];tc=-c[1];}
   else if(ay>ax&&ay>az){ma=ay;face=c[1]>0?2:3;sc=c[0];tc=c[1]>0?c[2]:-c[2];}
   else {check(az>ax&&az>ay,"nearest oracle avoids face ties");ma=az;face=c[2]>0?4:5;sc=c[2]>0?c[0]:-c[0];tc=-c[1];}
   unsigned n=4u>>mip;
   unsigned x=(unsigned)floorf((sc/ma+1.f)*.5f*(float)n);
   unsigned y=(unsigned)floorf((tc/ma+1.f)*.5f*(float)n);
   if(x>=n)x=n-1;if(y>=n)y=n-1;
   int cube=(int)floorf(c[3]+.5f);if(cube<0)cube=0;if(cube>=CUBES)cube=CUBES-1;
   for(unsigned k=0;k<4;k++)expected[k]=texel_byte((unsigned)cube,face,mip,x,y,k)?UINT32_C(0x3f800000):0;
}
static void draw_case(const Program *p,unsigned mode,unsigned filter,unsigned point,
                      const float coordinate[4],float lod,const float dx[3],const float dy[3],int has_oracle)
{
   glUseProgram(p->program);glUniform4fv(p->coordinate,1,coordinate);
   glUniform3fv(p->dx,1,dx);glUniform3fv(p->dy,1,dy);
   if(p->lod>=0)glUniform1f(p->lod,lod);
   const GLuint clear[4]={0xdeadbeef,0xdeadbeef,0xdeadbeef,0xdeadbeef};glClearBufferuiv(GL_COLOR,0,clear);
   check_gl("before CubeArray draw");
   printf("CASE %u mode=%u filter=%u point=%u coord=%08x,%08x,%08x,%08x lod=%08x dx=%08x,%08x,%08x dy=%08x,%08x,%08x oracle=%d\n",
      draws,mode,filter,point,bits(coordinate[0]),bits(coordinate[1]),bits(coordinate[2]),bits(coordinate[3]),bits(lod),
      bits(dx[0]),bits(dx[1]),bits(dx[2]),bits(dy[0]),bits(dy[1]),bits(dy[2]),has_oracle);
   fflush(stdout);fprintf(stderr,"DRAW: %u mode=%u filter=%u point=%u\n",draws,mode,filter,point);
   glDrawArrays(GL_TRIANGLES,0,3);glFinish();check_gl("CubeArray draw/finish");
   uint32_t raw[GUARD+PIXELS*4+GUARD],expected[4];
   for(unsigned i=0;i<GUARD+PIXELS*4+GUARD;i++)raw[i]=UINT32_C(0xa5a5a5a5);
   glReadPixels(0,0,SIDE,SIDE,GL_RGBA_INTEGER,GL_UNSIGNED_INT,raw+GUARD);check_gl("RGBA32UI readback");
   for(unsigned i=0;i<GUARD;i++)check(raw[i]==UINT32_C(0xa5a5a5a5)&&raw[GUARD+PIXELS*4+i]==UINT32_C(0xa5a5a5a5),"output guards");
   if(has_oracle){nearest_oracle(coordinate,mode?(unsigned)lod:0,expected);++oracle_draws;}
   for(unsigned pixel=0;pixel<PIXELS;pixel++)for(unsigned channel=0;channel<4;channel++){
      uint32_t value=raw[GUARD+pixel*4+channel];
      printf("WORD %u %u %u %08x\n",draws,pixel,channel,value);
      check(value<=UINT32_C(0x3f800000),"finite nonnegative normalized texture value");
      if(has_oracle){
         if(value!=expected[channel])fprintf(stderr,"MISMATCH draw=%u pixel=%u channel=%u actual=%08x expected=%08x\n",draws,pixel,channel,value,expected[channel]);
         check(value==expected[channel],"independent nearest face/cube/mip/texel oracle");
      }
   }
   printf("DONE %u\n",draws++);fflush(stdout);
}
int main(void)
{
   EGLDisplay display=eglGetDisplay(EGL_DEFAULT_DISPLAY);
   check(display!=EGL_NO_DISPLAY&&eglInitialize(display,NULL,NULL),"EGL initialize");
   const EGLint ca[]={EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
   const EGLint sa[]={EGL_WIDTH,1,EGL_HEIGHT,1,EGL_NONE};
   const EGLint xa[]={EGL_CONTEXT_MAJOR_VERSION,3,EGL_CONTEXT_MINOR_VERSION,1,EGL_NONE};
   EGLConfig config=NULL;EGLint count=0;
   check(eglBindAPI(EGL_OPENGL_ES_API)&&eglChooseConfig(display,ca,&config,1,&count)&&count==1,"EGL config");
   EGLSurface surface=eglCreatePbufferSurface(display,config,sa);
   EGLContext context=eglCreateContext(display,config,EGL_NO_CONTEXT,xa);
   check(surface!=EGL_NO_SURFACE&&context!=EGL_NO_CONTEXT&&eglMakeCurrent(display,surface,surface,context),"EGL current");
   fprintf(stderr,"RENDERER: %s\nVERSION: %s\n",glGetString(GL_RENDERER),glGetString(GL_VERSION));
   GLint extensions=0;int cube_array=0;glGetIntegerv(GL_NUM_EXTENSIONS,&extensions);
   for(GLint i=0;i<extensions;i++)if(!strcmp((const char*)glGetStringi(GL_EXTENSIONS,(GLuint)i),"GL_EXT_texture_cube_map_array"))cube_array=1;
   if(!cube_array){fprintf(stderr,"UNSUPPORTED: public GL_EXT_texture_cube_map_array\n");return 2;}
   check_gl("public CubeArray extension");
   Program programs[2]={make_program(0),make_program(1)};
   GLuint input,output,fbo[2],vao,vbo;glGenTextures(1,&input);glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,input);
   glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,LEVELS,GL_RGBA8,4,4,CUBES*FACES);
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_BASE_LEVEL,0);
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_MAX_LEVEL,LEVELS-1);
   glTexParameterf(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_MIN_LOD,0.f);
   glTexParameterf(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_MAX_LOD,1000.f);
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_WRAP_R,GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_COMPARE_MODE,GL_NONE);
   glPixelStorei(GL_PACK_ALIGNMENT,1);glPixelStorei(GL_UNPACK_ALIGNMENT,1);
   glGenFramebuffers(2,fbo);glBindFramebuffer(GL_FRAMEBUFFER,fbo[1]);
   const GLenum color=GL_COLOR_ATTACHMENT0;glDrawBuffers(1,&color);glReadBuffer(color);
   for(unsigned mip=0;mip<LEVELS;mip++){
      unsigned n=4u>>mip;unsigned char upload[4*4*CUBES*FACES*4];
      for(unsigned layer=0;layer<CUBES*FACES;layer++)for(unsigned y=0;y<n;y++)for(unsigned x=0;x<n;x++)for(unsigned k=0;k<4;k++)
         upload[((layer*n+y)*n+x)*4+k]=(unsigned char)texel_byte(layer/FACES,layer%FACES,mip,x,y,k);
      glTexSubImage3D(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,(GLint)mip,0,0,0,(GLsizei)n,(GLsizei)n,CUBES*FACES,GL_RGBA,GL_UNSIGNED_BYTE,upload);
      check_gl("CubeArray source upload");
      for(unsigned layer=0;layer<CUBES*FACES;layer++){
         glFramebufferTextureLayer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,input,(GLint)mip,(GLint)layer);
         check(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE,"source face/mip FBO");
         unsigned char actual[GUARD+4*4*4+GUARD];memset(actual,0xa5,sizeof(actual));
         glReadPixels(0,0,(GLsizei)n,(GLsizei)n,GL_RGBA,GL_UNSIGNED_BYTE,actual+GUARD);check_gl("source exact readback");
         for(unsigned i=0;i<GUARD;i++)check(actual[i]==0xa5&&actual[GUARD+n*n*4+i]==0xa5,"source guards");
         for(unsigned pixel=0;pixel<n*n;pixel++){
            const unsigned char *v=actual+GUARD+pixel*4;
            for(unsigned k=0;k<4;k++)check(v[k]==upload[(layer*n*n+pixel)*4+k],"source byte oracle");
            printf("UPLOAD %u %u %u %02x%02x%02x%02x\n",mip,layer,pixel,v[0],v[1],v[2],v[3]);
         }
      }
   }
   glFramebufferTextureLayer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,0,0,0);
   glGenTextures(1,&output);glBindTexture(GL_TEXTURE_2D,output);glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA32UI,SIDE,SIDE);
   glBindFramebuffer(GL_FRAMEBUFFER,fbo[0]);glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,output,0);
   glDrawBuffers(1,&color);glReadBuffer(color);check(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE,"output FBO");
   const float vertices[]={-1,-1,0,1,3,-1,0,1,-1,3,0,1};
   glGenVertexArrays(1,&vao);glBindVertexArray(vao);glGenBuffers(1,&vbo);glBindBuffer(GL_ARRAY_BUFFER,vbo);
   glBufferData(GL_ARRAY_BUFFER,sizeof(vertices),vertices,GL_STATIC_DRAW);glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,NULL);glEnableVertexAttribArray(0);
   glViewport(0,0,SIDE,SIDE);glDisable(GL_BLEND);glDisable(GL_CULL_FACE);glDisable(GL_DITHER);glDisable(GL_SCISSOR_TEST);glDisable(GL_DEPTH_TEST);
   const float zero[3]={0,0,0},layers[4]={-.75f,.49f,.51f,2.25f};
   for(unsigned filter=0;filter<2;filter++){
      glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_MIN_FILTER,filter?GL_LINEAR_MIPMAP_LINEAR:GL_NEAREST_MIPMAP_NEAREST);
      glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY_EXT,GL_TEXTURE_MAG_FILTER,filter?GL_LINEAR:GL_NEAREST);
      for(unsigned mode=0;mode<2;mode++)for(unsigned l=0;l<(filter?2u:4u);l++)for(unsigned point=0;point<(filter?18u:12u);point++)for(unsigned m=0;m<(mode?3u:1u);m++){
         float c[4];
         if(point<12)direction(point%6,point<6?0.f:.5f,point<6?0.f:-.5f,c);
         else {const float seam[6][3]={{1,0,1},{-1,0,-1},{0,1,1},{0,-1,-1},{1,1,1},{-1,-1,-1}};memcpy(c,seam[point-12],3*sizeof(float));}
         c[3]=filter?(float)l:layers[l];float lod=filter?(m==1?.5f:(float)m):(float)m;
         draw_case(&programs[mode],mode,filter,point,c,lod,zero,zero,!filter);
      }
   }
   /* Additional spatially varying implicit sampling, not merely uniform rays.
    * The interpolated pixel_position changes by1 per output pixel; face-tangent
    * gradients span >1 source texel/pixel and exercise nonzero implicit LOD. */
   for(unsigned cube=0;cube<CUBES;cube++)for(unsigned face=0;face<FACES;face++){
      float c[4],sx[3],sy[3],dx[3],dy[3];direction(face,0,0,c);c[3]=(float)cube;
      direction(face,1,0,sx);direction(face,0,1,sy);
      for(unsigned k=0;k<3;k++){dx[k]=(sx[k]-c[k])*(cube?1.5f:.75f);dy[k]=(sy[k]-c[k])*(cube?1.5f:.75f);}
      draw_case(&programs[0],2,1,face,c,0,dx,dy,0);
   }
   check(draws==348&&oracle_draws==192,"complete case inventory");
   glDeleteProgram(programs[0].program);glDeleteProgram(programs[1].program);
   glDeleteBuffers(1,&vbo);glDeleteVertexArrays(1,&vao);glDeleteFramebuffers(2,fbo);glDeleteTextures(1,&input);glDeleteTextures(1,&output);check_gl("cleanup");
   printf("PASS draws=%u checks=%u nearest_oracle=%u observations=%u\n",draws,checks,oracle_draws,draws-oracle_draws);fflush(stdout);
   eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);eglDestroyContext(display,context);eglDestroySurface(display,surface);eglTerminate(display);return 0;
}
