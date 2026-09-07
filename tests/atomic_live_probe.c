/* SPDX-License-Identifier: MIT */
/* Real EGL/GLES 3.1 SSBO atomic A/B probe. The GPU produces every result.
 * stdout contains observed DWORDs, never substituted host answers. Host checks
 * are ordinary test oracles: deterministic scalar semantics and the existence
 * of a legal serialization of concurrent return-old transitions. Different
 * backends need not choose the same invocation/workgroup order.
 *
 * Compile standalone against the selected Mesa EGL/GLES headers and libraries.
 * No renderer-name dispatch, shader recognition, or prepared GPU responses. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_LANES = 128, OP_COUNT = 10 };
enum operation { ADD, SMIN, SMAX, UMIN, UMAX, AND, OR, XOR, EXCHANGE, CMP_SWAP };
static const char *const names[] = {
   "add", "signed_min", "signed_max", "unsigned_min", "unsigned_max",
   "and", "or", "xor", "exchange", "compare_swap"
};
static const char *const builtins[] = {
   "atomicAdd", "atomicMin", "atomicMax", "atomicMin", "atomicMax",
   "atomicAnd", "atomicOr", "atomicXor", "atomicExchange", "atomicCompSwap"
};
static unsigned failures, scenarios, dispatches, checks;
static size_t unit_words;

static void
check(int condition, const char *name, enum operation op, unsigned scenario)
{
   ++checks;
   if (!condition) {
      ++failures;
      fprintf(stderr, "INVARIANT_FAIL\toperation=%s\tscenario=%u\tcheck=%s\n",
              names[op], scenario, name);
   }
}

static int
check_gl(const char *where)
{
   int ok = 1;
   GLenum error;
   while ((error = glGetError()) != GL_NO_ERROR) {
      fprintf(stderr, "GL_ERROR\t%s\t0x%x\n", where, error);
      ++failures;
      ok = 0;
   }
   return ok;
}

static uint32_t
apply_atomic(enum operation op, uint32_t old, uint32_t value, uint32_t compare)
{
   /* Unsigned C arithmetic defines wraparound. XORing the sign bit orders
    * signed two's-complement values without implementation-defined casts. */
   switch (op) {
   case ADD: return old + value;
   case SMIN: return (old ^ UINT32_C(0x80000000)) < (value ^ UINT32_C(0x80000000)) ? old : value;
   case SMAX: return (old ^ UINT32_C(0x80000000)) > (value ^ UINT32_C(0x80000000)) ? old : value;
   case UMIN: return old < value ? old : value;
   case UMAX: return old > value ? old : value;
   case AND: return old & value;
   case OR: return old | value;
   case XOR: return old ^ value;
   case EXCHANGE: return value;
   case CMP_SWAP: return old == compare ? value : old;
   }
   abort();
}

static GLuint
make_program(enum operation op, unsigned local_size)
{
   char source[4096], expression_a[160], expression_b[160];
   const char *type = op == SMIN || op == SMAX ? "int" : "uint";
   if (op == CMP_SWAP) {
      snprintf(expression_a, sizeof(expression_a), "%s(a[alias_index], cmp, arg)", builtins[op]);
      snprintf(expression_b, sizeof(expression_b), "%s(b[2], cmp, arg)", builtins[op]);
   } else {
      snprintf(expression_a, sizeof(expression_a), "%s(a[alias_index], %s(arg))", builtins[op], type);
      snprintf(expression_b, sizeof(expression_b), "%s(b[2], %s(arg))", builtins[op], type);
   }
   snprintf(source, sizeof(source),
      "#version 310 es\nprecision highp int;\n"
      "layout(local_size_x=%u) in;\n"
      "layout(std430,binding=0) coherent buffer A { %s a[]; };\n"
      "layout(std430,binding=1) coherent buffer B { %s b[]; };\n"
      "layout(std430,binding=2) writeonly buffer Out { uint output_words[]; };\n"
      "layout(std430,binding=3) readonly buffer In { uvec2 operand[]; };\n"
      "uniform highp uint active_limit, sparse, alias_index, alias_choice;\n"
      "void main(){\n"
      " uint id=gl_GlobalInvocationID.x;\n"
      " if(id>=active_limit || (sparse!=0u && (id&3u)==1u)) return;\n"
      " uint arg=operand[id].x, cmp=operand[id].y, old;\n"
      " if(((id+alias_choice)&1u)==0u) old=uint(%s);\n"
      " else old=uint(%s);\n"
      " output_words[3u*id]=old;\n"
      " output_words[3u*id+1u]=arg;\n"
      " output_words[3u*id+2u]=0xc0000000u|id;\n}\n",
      local_size, type, type, expression_a, expression_b);
   GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
   GLuint program = 0;
   const char *pointer = source;
   GLint ok = 0;
   glShaderSource(shader, 1, &pointer, NULL);
   glCompileShader(shader);
   glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[8192];
      glGetShaderInfoLog(shader, sizeof(log), NULL, log);
      fprintf(stderr, "SHADER_ERROR\toperation=%s\tlocal_size=%u\n%s\nSOURCE\n%s\n",
              names[op], local_size, log, source);
      ++failures;
      goto done;
   }
   program = glCreateProgram();
   glAttachShader(program, shader);
   glLinkProgram(program);
   glGetProgramiv(program, GL_LINK_STATUS, &ok);
   if (!ok) {
      char log[8192];
      glGetProgramInfoLog(program, sizeof(log), NULL, log);
      fprintf(stderr, "LINK_ERROR\toperation=%s\tlocal_size=%u\n%s\n",
              names[op], local_size, log);
      ++failures;
      glDeleteProgram(program);
      program = 0;
   }
done:
   glDeleteShader(shader);
   check_gl("make_program");
   return program;
}

static int
read_buffer(GLuint buffer, uint32_t *words, size_t count)
{
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
   const void *mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                         (GLsizeiptr)(count * 4), GL_MAP_READ_BIT);
   if (!mapped) {
      fprintf(stderr, "MAP_ERROR\tbuffer=%u\n", buffer);
      ++failures;
      check_gl("map_buffer");
      return 0;
   }
   memcpy(words, mapped, count * 4);
   if (!glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)) {
      fprintf(stderr, "UNMAP_ERROR\tbuffer=%u\n", buffer);
      ++failures;
      return 0;
   }
   return check_gl("read_buffer");
}

static void
emit(const char *mode, enum operation op, unsigned scenario, const char *field,
     unsigned lane, uint32_t word)
{
   printf("%s\t%s\t%u\t%s\t%u\t%08" PRIx32 "\n",
          mode, names[op], scenario, field, lane, word);
}

/* Hierholzer's algorithm checks that the observed old->new edges have one
 * complete directed Euler trail, starting at the uploaded value and ending at
 * the read-back value. This is a necessary and sufficient serialization for
 * these one-atomic-per-active-invocation shaders, including no-op/self-loops.
 * Checking only a final sum or set of old values would miss several failures. */
static int
legal_serialization(uint32_t initial, uint32_t final, unsigned count,
                    const uint32_t *old, const uint32_t *next)
{
   unsigned char used[MAX_LANES] = {0};
   uint32_t stack[MAX_LANES + 1], reverse[MAX_LANES + 1];
   unsigned top = 1, path_count = 0, consumed = 0;
   stack[0] = initial;
   while (top) {
      unsigned edge;
      for (edge = 0; edge < count; ++edge)
         if (!used[edge] && old[edge] == stack[top - 1])
            break;
      if (edge == count) {
         reverse[path_count++] = stack[--top];
      } else {
         used[edge] = 1;
         ++consumed;
         stack[top++] = next[edge];
      }
   }
   if (consumed != count || path_count != count + 1 ||
       reverse[0] != final || reverse[count] != initial)
      return 0;
   /* Unbalanced directed edges can all be popped without making a trail.
    * Verify the reconstructed path consumes the original edge multiset. */
   memset(used, 0, sizeof(used));
   for (unsigned index = count; index > 0; --index) {
      unsigned edge;
      for (edge = 0; edge < count; ++edge)
         if (!used[edge] && old[edge] == reverse[index] && next[edge] == reverse[index - 1])
            break;
      if (edge == count)
         return 0;
      used[edge] = 1;
   }
   return 1;
}

static void
run_scenario(GLuint program, enum operation op, unsigned scenario,
             unsigned local_size, unsigned groups, unsigned active_limit,
             unsigned sparse, uint32_t initial, uint32_t value, uint32_t compare)
{
   const unsigned lanes = local_size * groups;
   const size_t words = 4 * unit_words;
   const size_t target = 2 * unit_words + 2;
   const size_t bytes = words * 4;
   const unsigned before = failures;
   const char *mode = lanes == 1 ? "deterministic" : "contention";
   uint32_t *upload[3] = {NULL, NULL, NULL}, *actual[3] = {NULL, NULL, NULL};
   GLuint buffers[3] = {0, 0, 0};
   unsigned active = 0;
   uint32_t edges_old[MAX_LANES], edges_new[MAX_LANES];
   if (lanes > MAX_LANES || active_limit > lanes) {
      ++failures;
      return;
   }
   ++scenarios;
   for (unsigned buffer = 0; buffer < 3; ++buffer) {
      upload[buffer] = malloc(bytes);
      actual[buffer] = malloc(bytes);
      if (!upload[buffer] || !actual[buffer]) {
         ++failures;
         goto cleanup;
      }
      for (size_t word = 0; word < words; ++word)
         upload[buffer][word] = UINT32_C(0x6a09e667) ^ (uint32_t)word * UINT32_C(0x9e3779b9) ^ (buffer << 24);
   }
   upload[0][target] = initial;
   for (unsigned lane = 0; lane < lanes; ++lane) {
      uint32_t arg = lanes == 1 ? value : UINT32_C(0x9e3779b9) * (lane + 1u);
      uint32_t cmp = compare;
      if (lanes > 1 && op == ADD)
         arg = 1;
      if (lanes > 1 && (op == EXCHANGE || op == CMP_SWAP))
         arg = UINT32_C(0x10000) + lane;
      if (lanes > 1 && op == CMP_SWAP)
         cmp = (lane & 1u) ? (initial ^ UINT32_C(0x01000000)) : initial;
      upload[2][unit_words + 2 * lane] = arg;
      upload[2][unit_words + 2 * lane + 1] = cmp;
   }
   glGenBuffers(3, buffers);
   for (unsigned buffer = 0; buffer < 3; ++buffer) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[buffer]);
      glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)bytes, upload[buffer], GL_DYNAMIC_COPY);
   }
   /* Same backing address through distinct overlapping, nonzero-offset views.
    * No restrict qualifier: both aliases must share one atomic memory location. */
   glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, buffers[0], (GLintptr)(unit_words * 4), (GLsizeiptr)(unit_words * 8));
   glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, buffers[0], (GLintptr)(unit_words * 8), (GLsizeiptr)(unit_words * 4));
   glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, buffers[1], (GLintptr)(unit_words * 4), (GLsizeiptr)(unit_words * 4));
   glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 3, buffers[2], (GLintptr)(unit_words * 4), (GLsizeiptr)(unit_words * 4));
   glUseProgram(program);
   glUniform1ui(glGetUniformLocation(program, "active_limit"), active_limit);
   glUniform1ui(glGetUniformLocation(program, "sparse"), sparse);
   glUniform1ui(glGetUniformLocation(program, "alias_index"), (GLuint)unit_words + 2);
   glUniform1ui(glGetUniformLocation(program, "alias_choice"), scenario & 1u);
   if (!check_gl("upload_and_bind"))
      goto cleanup;
   fprintf(stderr, "DISPATCH\toperation=%s\tscenario=%u\tlocal=%u\tgroups=%u\tactive_limit=%u\tsparse=%u\tinitial=%08" PRIx32 "\n",
           names[op], scenario, local_size, groups, active_limit, sparse, initial);
   glDispatchCompute(groups, 1, 1);
   ++dispatches;
   glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
   glFinish();
   if (!check_gl("dispatch_and_barrier"))
      goto cleanup;
   for (unsigned buffer = 0; buffer < 3; ++buffer)
      if (!read_buffer(buffers[buffer], actual[buffer], words))
         goto cleanup;
   unsigned bad_state = 0, bad_output = 0, bad_input = 0;
   for (size_t word = 0; word < words; ++word) {
      if (word != target && actual[0][word] != upload[0][word])
         ++bad_state;
      if (actual[2][word] != upload[2][word])
         ++bad_input;
      int written = 0;
      if (word >= unit_words && word < unit_words + lanes * 3) {
         unsigned lane = (unsigned)((word - unit_words) / 3);
         written = lane < active_limit && !(sparse && (lane & 3u) == 1u);
      }
      if (!written && actual[1][word] != upload[1][word])
         ++bad_output;
   }
   check(bad_state == 0, "aliased_state_sentinels", op, scenario);
   check(bad_input == 0, "readonly_input_and_sentinels", op, scenario);
   check(bad_output == 0, "inactive_lane_and_output_sentinels", op, scenario);
   emit(mode, op, scenario, "final", 0, actual[0][target]);
   for (unsigned lane = 0; lane < lanes; ++lane) {
      if (lane >= active_limit || (sparse && (lane & 3u) == 1u))
         continue;
      const uint32_t *result = actual[1] + unit_words + lane * 3;
      const uint32_t *args = upload[2] + unit_words + lane * 2;
      check(result[1] == args[0], "operand_transport", op, scenario);
      check(result[2] == (UINT32_C(0xc0000000) | lane), "active_lane_marker", op, scenario);
      emit(mode, op, scenario, "return_old", lane, result[0]);
      emit(mode, op, scenario, "operand", lane, result[1]);
      emit(mode, op, scenario, "active_marker", lane, result[2]);
      edges_old[active] = result[0];
      edges_new[active] = apply_atomic(op, result[0], args[0], args[1]);
      ++active;
   }
   check(legal_serialization(initial, actual[0][target], active, edges_old, edges_new),
         "complete_legal_atomic_serialization", op, scenario);
   if (lanes == 1) {
      check(edges_old[0] == initial, "deterministic_return_old", op, scenario);
      check(actual[0][target] == apply_atomic(op, initial, value, compare), "deterministic_final", op, scenario);
   }
   fprintf(stderr, "SCENARIO\toperation=%s\tscenario=%u\tmode=%s\tactive=%u\tchecks=%s\n",
           names[op], scenario, mode, active, before == failures ? "Pass" : "Fail");
cleanup:
   for (unsigned binding = 0; binding < 4; ++binding)
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, 0);
   glDeleteBuffers(3, buffers);
   for (unsigned buffer = 0; buffer < 3; ++buffer) {
      free(upload[buffer]);
      free(actual[buffer]);
   }
   check_gl("scenario_cleanup");
}

int
main(void)
{
   EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   EGLint major = 0, minor = 0, count = 0;
   EGLConfig config = NULL;
   EGLSurface surface = EGL_NO_SURFACE;
   EGLContext context = EGL_NO_CONTEXT;
   int current = 0, status = 2;
   if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) {
      fprintf(stderr, "EGL_INITIALIZE_ERROR\t0x%x\n", eglGetError());
      return 2;
   }
   const EGLint config_attributes[] = {
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE
   };
   const EGLint context_attributes[] = {
      EGL_CONTEXT_MAJOR_VERSION_KHR, 3, EGL_CONTEXT_MINOR_VERSION_KHR, 1, EGL_NONE
   };
   const EGLint surface_attributes[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
   if (!eglBindAPI(EGL_OPENGL_ES_API) ||
       !eglChooseConfig(display, config_attributes, &config, 1, &count) || count != 1)
      goto cleanup;
   surface = eglCreatePbufferSurface(display, config, surface_attributes);
   context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
   if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT ||
       !eglMakeCurrent(display, surface, surface, context))
      goto cleanup;
   current = 1;
   fprintf(stderr, "GL\tvendor=%s\trenderer=%s\tversion=%s\tglsl=%s\n",
           glGetString(GL_VENDOR), glGetString(GL_RENDERER),
           glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));
   GLint alignment = 0, local_limit = 0, storage_blocks = 0;
   glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &alignment);
   glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &storage_blocks);
   glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &local_limit);
   fprintf(stderr, "LIMITS\tssbo_offset_alignment=%d\tlocal_x=%d\tstorage_blocks=%d\n",
           alignment, local_limit, storage_blocks);
   if (!check_gl("context") || alignment <= 0 || alignment > 1048576 ||
       (alignment & 3) || local_limit < 37 || storage_blocks < 4)
      goto cleanup;
   unit_words = ((2048u + (size_t)alignment - 1) / (size_t)alignment) * (size_t)alignment / 4;
   puts("mode\toperation\tscenario\tfield\tlane\tu32_hex");
   static const uint32_t initial[] = {0xfffffffeu, 0x80000000u, 0x7fffffffu, 0x01234567u, 0xffffffffu, 0};
   static const uint32_t value[] = {5, 0x7fffffffu, 0x80000000u, 0xf0f00f0fu, 1, 0xffffffffu};
   for (enum operation op = ADD; op < OP_COUNT; op = (enum operation)(op + 1)) {
      GLuint program = make_program(op, 1);
      if (!program)
         continue;
      for (unsigned index = 0; index < 6; ++index)
         run_scenario(program, op, index, 1, 1, 1, 0, initial[index], value[index],
                      (index & 1u) ? initial[index] ^ 0x100u : initial[index]);
      glDeleteProgram(program);
      program = make_program(op, 30);
      if (program) {
         run_scenario(program, op, 6, 30, 3, 90, 0, 0xfffffff0u, 0, 0);
         glDeleteProgram(program);
      }
      program = make_program(op, 37);
      if (program) {
         run_scenario(program, op, 7, 37, 2, 74, 0, 0x80000000u, 0, 0);
         run_scenario(program, op, 8, 37, 3, 108, 1, 0x7fffffffu, 0, 0);
         glDeleteProgram(program);
      }
   }
   fprintf(stderr, "RESULT\tscenarios=%u\texpected_scenarios=90\tdispatches=%u\tchecks=%u\tfailures=%u\n",
           scenarios, dispatches, checks, failures);
   status = failures || scenarios != 90 || dispatches != 90 ? 1 : 0;
cleanup:
   if (current) {
      glUseProgram(0);
      check_gl("cleanup");
      if (failures && status == 0)
         status = 1;
      eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
   }
   if (context != EGL_NO_CONTEXT)
      eglDestroyContext(display, context);
   if (surface != EGL_NO_SURFACE)
      eglDestroySurface(display, surface);
   if (status == 2)
      fprintf(stderr, "SETUP_FAILED\tegl_error=0x%x\n", eglGetError());
   eglTerminate(display);
   return status;
}
