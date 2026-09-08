#!/usr/bin/env bash
# Real Mesa NIR -> native PCO -> raw/prepared ISS texture handshakes.
# Build only private test objects; never rebuild/install the shared runtime.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    source "${REPO_DIR}/config/local.env"
    set +a
fi
MESA_TEST_BUILD="${PVRGPU_MESA_PVRGPU_BUILD_DIR:-${PVRGPU_BUILD_DIR:?}/mesa-pvrgpu}"
MESA_TEST_OUT="$(mktemp -d "${PVRGPU_TMP_ROOT:-${TMPDIR:-/tmp}}/pvrgpu-texture-sequence.XXXXXX")"
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
objects = %w[src/gallium/drivers/pvrgpu/pvrgpu_pco.c tests/pvrgpu_texture_sequence_test.c].map do |source|
  object = File.join(output, File.basename(source) + '.o')
  abort 'C compilation failed' unless system(compiler, *flags, '-c', File.join(repo, source),
    '-o', object, chdir: entry.fetch('directory'))
  object
end
ninja = File.read(File.join(build, 'build.ninja'))
line = ninja[/build src\/gallium\/drivers\/pvrgpu\/pvrgpu_pco_lowering_test:.*\n LINK_ARGS = ([^\n]+)/, 1]
abort 'Mesa PCO test link command unavailable' unless line
cpp = database.find { |item| File.extname(item.fetch('file')) == '.cpp' }
cxx = (cpp['arguments'] || Shellwords.split(cpp.fetch('command'))).first
compiler_test = File.join(output, 'compiler-test')
abort 'compiler link failed' unless system(cxx, *objects, *Shellwords.split(line),
  '-o', compiler_test, chdir: build)
abort 'native sequence compilation failed' unless system(compiler_test, output)
native_test = File.join(output, 'native-test')
abort 'native consumer compilation failed' unless system(cxx, '-std=c++17', '-O1',
  '-fsanitize=address,undefined,float-cast-overflow', '-fno-omit-frame-pointer',
  '-I' + File.join(repo, 'src/systemc'), File.join(repo, 'tests/pco_texture_sequence_test.cpp'),
  File.join(repo, 'src/systemc/shader/pco_iss.cpp'), '-o', native_test)
abort 'native sequence checks failed' unless system({'ASAN_OPTIONS' => 'detect_leaks=0',
  'UBSAN_OPTIONS' => 'halt_on_error=1'}, native_test, output)
puts "Native texture sequence ASan/UBSan artifacts: #{output}"
RUBY
