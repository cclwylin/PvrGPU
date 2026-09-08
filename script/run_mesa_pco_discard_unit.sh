#!/usr/bin/env bash
# Compile literal GLSL with the actual screen options, then replay its native
# discard/texture/derivative program. All artifacts are private, no Ninja build.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    source "${REPO_DIR}/config/local.env"
    set +a
fi
MESA_TEST_BUILD="${PVRGPU_MESA_PVRGPU_BUILD_DIR:-${PVRGPU_BUILD_DIR:?}/mesa-pvrgpu}"
MESA_TEST_OUT="$(mktemp -d "${PVRGPU_TMP_ROOT:-${TMPDIR:-/tmp}}/pvrgpu-discard.XXXXXX")"
export REPO_DIR MESA_TEST_BUILD MESA_TEST_OUT
ruby -rjson -rshellwords <<'RUBY'
repo, build, output = ENV.values_at('REPO_DIR', 'MESA_TEST_BUILD', 'MESA_TEST_OUT')
puts "Discard artifacts: #{output}"
database = JSON.parse(File.read(File.join(build, 'compile_commands.json')))
compile = lambda do |source, template|
  entry = database.find { |item| File.basename(item.fetch('file')) == template }
  abort "No Mesa compiler command for #{template}" unless entry
  command = entry['arguments'] || Shellwords.split(entry.fetch('command'))
  flags = []
  until command.empty?
    arg = command.shift
    if %w[-o -MF -MQ -MT].include?(arg)
      command.shift
    elsif %w[-c -MD -MMD -g].include?(arg) || File.basename(arg) == template
      next
    else
      flags << arg
    end
  end
  object = File.join(output, File.basename(source) + '.o')
  abort "Compile failed: #{source}" unless system(*flags,
    '-I' + File.join(repo, 'src/gallium/drivers/pvrgpu'), '-O1', '-c', File.join(repo, source),
    '-o', object, chdir: entry.fetch('directory'))
  object
end
objects = [compile.call('src/gallium/drivers/pvrgpu/pvrgpu_pco.c', 'pvrgpu_pco.c'),
           compile.call('tests/pvrgpu_fragment_discard_options_test.c', 'pvrgpu_screen.c'),
           compile.call('tests/pvrgpu_fragment_discard_test.cpp', 'standalone_scaffolding.cpp')]
ninja = File.read(File.join(build, 'build.ninja'))
link = ninja[/build src\/gallium\/drivers\/pvrgpu\/pvrgpu_pco_lowering_test:.*\n LINK_ARGS = ([^\n]+)/, 1]
abort 'Mesa PCO link command unavailable' unless link
cpp = database.find { |item| File.basename(item.fetch('file')) == 'standalone_scaffolding.cpp' }
cxx = (cpp['arguments'] || Shellwords.split(cpp.fetch('command'))).first
compiler = File.join(output, 'compiler-test')
abort 'Compiler test link failed' unless system(cxx, *objects,
  'src/compiler/glsl/libglsl_standalone.a', 'src/compiler/glsl/libglsl.a',
  'src/compiler/glsl/glcpp/libglcpp.a', 'src/compiler/glsl/libglsl_util.a',
  'src/compiler/glsl/glcpp/libglcpp_standalone.a',
  *Shellwords.split(link), '-Wl,-dead_strip', '-o', compiler, chdir: build)
fixtures = %w[conditional.pco unconditional.pco].map { |name| File.join(output, name) }
abort 'GLSL discard compiler test failed' unless system(compiler, *fixtures)
native = File.join(output, 'native-test')
abort 'Native discard test compilation failed' unless system(cxx, '-std=c++17', '-O1',
  '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
  '-I' + File.join(repo, 'src/systemc'), File.join(repo, 'tests/pco_fragment_discard_test.cpp'),
  File.join(repo, 'src/systemc/shader/pco_iss.cpp'), '-o', native)
abort 'Native discard continuation failed' unless system(
  {'ASAN_OPTIONS' => 'detect_leaks=0', 'UBSAN_OPTIONS' => 'halt_on_error=1'}, native, *fixtures)
RUBY
