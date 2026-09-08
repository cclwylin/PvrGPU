// SPDX-License-Identifier: MIT
// Literal GLSL -> actual Gallium frontend options -> NIR -> native PCO.
#include "main/mtypes.h"
#include "compiler/glsl/glsl_parser_extras.h"
#include "compiler/glsl/standalone_scaffolding.h"
#include "compiler/glsl/builtin_functions.h"
#include "compiler/glsl/linker_util.h"
#include "nir/nir.h"
#include "compiler/glsl/gl_nir_linker.h"
#include "pipe/p_screen.h"
extern "C" {
#include "pvrgpu_pco.h"
const nir_shader_compiler_options *pvrgpu_discard_test_options(void);
}
#include <cstdio>
#include <cstdlib>
#include <cstring>

static unsigned checks;
static void Check(bool ok, const char *why)
{
   ++checks;
   if (!ok) { std::fprintf(stderr, "%s\n", why); std::exit(1); }
}
static unsigned Count(nir_shader *nir, nir_intrinsic_op op)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, nir)
      nir_foreach_block(block, impl)
         nir_foreach_instr(instr, block)
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op) ++count;
   return count;
}
static gl_shader_program *Compile(gl_context *ctx,
   const nir_shader_compiler_options *options, bool unconditional)
{
   initialize_context_to_defaults(ctx, API_OPENGLES2);
   ctx->Version = 31;
   ctx->Const.GLSLVersion = 310;
   ctx->Const.MaxUserAssignableUniformLocations = 4096;
   ctx->Const.Program[MESA_SHADER_VERTEX].MaxCombinedUniformComponents = 1024;
   ctx->Const.Program[MESA_SHADER_FRAGMENT].MaxCombinedUniformComponents = 1024;
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage)
      ctx->screen->nir_options[stage] = options;
   auto *program = standalone_create_shader_program();
   program->IsES = true;
   const char *source[] = {
      "#version 310 es\nlayout(location=0) in highp vec4 position;\n"
      "void main() { gl_Position = position; }\n",
      unconditional ?
      "#version 310 es\nprecision highp float;\nlayout(location=0) out vec4 color;\n"
      "void main() { discard; color=vec4(1.0); }\n" :
      "#version 310 es\nprecision highp float;\n"
      "layout(binding=0) uniform highp sampler2D maskTex;\n"
      "layout(binding=1) uniform highp sampler2D colorTex;\n"
      "layout(location=0) out vec4 color;\n"
      "void main() { vec4 mask=texture(maskTex,vec2(0.25,0.75));\n"
      "if (mask.x < 0.5) discard;\n"
      "vec4 value=texture(colorTex,vec2(0.25,0.75));\n"
      "color=vec4(value.x,dFdx(mask.x),dFdy(mask.x),1.0); }\n"
   };
   for (unsigned stage = 0; stage < 2; ++stage) {
      auto *shader = standalone_add_shader_source(ctx, program,
         stage ? GL_FRAGMENT_SHADER : GL_VERTEX_SHADER, source[stage]);
      _mesa_glsl_compile_shader(ctx, shader, nullptr, false, false, true);
      if (shader->CompileStatus != COMPILE_SUCCESS)
         std::fprintf(stderr, "%s\n", shader->InfoLog);
      Check(shader->CompileStatus == COMPILE_SUCCESS, "literal GLSL compilation failed");
   }
   link_shaders_init(ctx, program);
   gl_nir_link_glsl(ctx, program);
   if (program->data->LinkStatus != LINKING_SUCCESS)
      std::fprintf(stderr, "%s\n", program->data->InfoLog);
   Check(program->data->LinkStatus == LINKING_SUCCESS, "literal GLSL link failed");
   return program;
}

int main(int argc, char **argv)
{
   Check(argc == 3, "expected conditional and unconditional native output paths");
   glsl_type_singleton_init_or_ref();
   _mesa_glsl_builtin_functions_init_or_ref();
   const auto *actual = pvrgpu_discard_test_options();
   Check(actual && actual->discard_is_demote,
         "actual Gallium fragment frontend must advertise discard_is_demote");
   char error[1024] = {};
   auto *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   Check(compiler, error);
   for (unsigned variant = 0; variant < 4; ++variant) {
      const bool correct = variant < 2, unconditional = variant & 1;
      auto control = *actual;
      control.discard_is_demote = false;
      gl_context ctx;
      auto *program = Compile(&ctx, correct ? actual : &control, unconditional);
      auto *vs = program->_LinkedShaders[MESA_SHADER_VERTEX]->Program->nir;
      auto *fs = program->_LinkedShaders[MESA_SHADER_FRAGMENT]->Program->nir;
      const unsigned demotes = Count(fs, nir_intrinsic_demote) + Count(fs, nir_intrinsic_demote_if);
      const unsigned terminates = Count(fs, nir_intrinsic_terminate) + Count(fs, nir_intrinsic_terminate_if);
      Check(correct ? demotes != 0 && terminates == 0 : terminates != 0 && demotes == 0,
            "GLSL discard did not follow the actual frontend compiler option");
      if (correct)
         Check(Count(fs, unconditional ? nir_intrinsic_demote : nir_intrinsic_demote_if) == 1,
               "literal GLSL must cover both unconditional demote and demote_if validation");
      std::printf("GLSL %s option=%u demote=%u terminate=%u\n",
         unconditional ? "unconditional" : "conditional", correct, demotes, terminates);
      if (correct) {
         const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
         pvrgpu_pco_graphics_binary binary = {};
         const bool compiled = pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, false, 1, 0, 0, 1, unconditional ? 0 : 2,
            &binary, error, sizeof(error));
         if (!compiled) { nir_print_shader(vs, stderr); nir_print_shader(fs, stderr); }
         Check(compiled, error[0] ? error : "PCO compiler rejected GLSL NIR without a reason");
         FILE *file = std::fopen(argv[unconditional ? 2 : 1], "wb");
         Check(file != nullptr, "open native discard fixture");
         Check(std::fwrite(binary.fragment.data, 1, binary.fragment.size, file) == binary.fragment.size,
               "write actual native discard bytes");
         Check(std::fclose(file) == 0, "close native discard fixture");
         std::printf("native bytes=%zu shared=%u push=%u/%u coefficients=%u\n",
            binary.fragment.size, binary.fragment.abi.shareds,
            binary.fragment.abi.push_constant_start, binary.fragment.abi.push_constant_count,
            binary.fragment.abi.coefficients);
         pvrgpu_pco_graphics_binary_finish(&binary);
      }
      standalone_destroy_shader_program(program);
      std::free(ctx.screen);
   }
   pvrgpu_pco_compiler_destroy(compiler);
   _mesa_glsl_builtin_functions_decref();
   glsl_type_singleton_decref();
   std::printf("GLSL discard frontend/compiler: PASS (%u checks)\n", checks);
}
