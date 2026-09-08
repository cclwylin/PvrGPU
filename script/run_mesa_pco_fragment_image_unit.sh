#!/usr/bin/env bash
# Isolated real Mesa FS image compiler + USC ISS + modeled GPU memory tests.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    source "${REPO_DIR}/config/local.env"
    set +a
fi
MESA_TEST_BUILD="${PVRGPU_MESA_PVRGPU_BUILD_DIR:-${PVRGPU_BUILD_DIR:?}/mesa-pvrgpu}"
MESA_TEST_OUT="$(mktemp -d "${PVRGPU_TMP_ROOT:-${TMPDIR:-/tmp}}/pvrgpu-fragment-image.XXXXXX")"
export REPO_DIR MESA_TEST_BUILD MESA_TEST_OUT
ruby -rjson -rshellwords <<'RUBY'
repo, build, output = ENV.values_at('REPO_DIR', 'MESA_TEST_BUILD', 'MESA_TEST_OUT')
database = JSON.parse(File.read(File.join(build, 'compile_commands.json')))
entry = database.find { |item| File.basename(item.fetch('file')) == 'pvrgpu_pco.c' }
abort 'Mesa PCO compile command unavailable' unless entry
command = entry['arguments'] || Shellwords.split(entry.fetch('command'))
compiler = command.shift
flags = ['-I' + File.join(repo, 'src/gallium/drivers/pvrgpu')]
until command.empty?
  arg = command.shift
  if %w[-o -MF -MQ -MT].include?(arg)
    command.shift
  elsif %w[-c -MD -MMD -g].include?(arg) || File.basename(arg) == 'pvrgpu_pco.c'
    next
  else
    flags << arg
  end
end
objects = %w[src/gallium/drivers/pvrgpu/pvrgpu_pco.c tests/pvrgpu_fragment_image_compiler_test.c].map do |source|
  object = File.join(output, File.basename(source) + '.o')
  abort 'C compilation failed' unless system(compiler, *flags, '-c', File.join(repo, source),
    '-o', object, chdir: entry.fetch('directory'))
  object
end
ninja = File.read(File.join(build, 'build.ninja'))
link_line = ninja[/build src\/gallium\/drivers\/pvrgpu\/pvrgpu_pco_lowering_test:.*\n LINK_ARGS = ([^\n]+)/, 1]
abort 'Mesa PCO test link command unavailable' unless link_line
cpp = database.find { |item| File.extname(item.fetch('file')) == '.cpp' }
cxx = (cpp['arguments'] || Shellwords.split(cpp.fetch('command'))).first
sources = %w[tests/pco_fragment_image_test.cpp src/systemc/shader/pco_iss.cpp
  src/systemc/memory/gpu_memory_system.cpp src/systemc/memory/dram_address_space.cpp
  src/systemc/cache_mmu/cache_array.cpp]
executable = File.join(output, 'native-fragment-image-test')
abort 'native compiler/ISS link failed' unless system(cxx, '-std=c++17', '-O1',
  '-I' + File.join(repo, 'model_stub'), '-I' + File.join(repo, 'src'),
  '-I' + File.join(repo, 'src/systemc'), *sources.map { |path| File.join(repo, path) },
  *objects, *Shellwords.split(link_line), '-o', executable, chdir: build)
abort 'native fragment image test failed' unless system(executable)
puts "Native fragment image unit artifacts: #{output}"
RUBY
