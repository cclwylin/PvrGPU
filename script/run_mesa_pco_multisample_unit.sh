#!/usr/bin/env bash
# Isolated native fixture regeneration and ISS validation. Never installs or
# rebuilds a shared Mesa/SystemC runtime used by live dEQP jobs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    source "${REPO_DIR}/config/local.env"
    set +a
fi
MESA_TEST_BUILD="${PVRGPU_MESA_PVRGPU_BUILD_DIR:-${PVRGPU_BUILD_DIR:?}/mesa-pvrgpu}"
MESA_TEST_OUT="$(mktemp -d "${PVRGPU_TMP_ROOT:-${TMPDIR:-/tmp}}/pvrgpu-ms-texture.XXXXXX")"
export REPO_DIR MESA_TEST_BUILD MESA_TEST_OUT
ruby -rjson -rshellwords <<'RUBY'
repo, build, output = ENV.values_at('REPO_DIR', 'MESA_TEST_BUILD', 'MESA_TEST_OUT')
database = JSON.parse(File.read(File.join(build, 'compile_commands.json')))
entry = database.find { |item| File.basename(item.fetch('file')) == 'pvrgpu_pco.c' }
abort 'Mesa PCO compile command unavailable' unless entry
command = entry['arguments'] || Shellwords.split(entry.fetch('command'))
compiler = command.shift
flags = []
until command.empty?
  arg = command.shift
  if %w[-o -MF -MQ -MT].include?(arg)
    command.shift
  elsif %w[-c -MD -MMD].include?(arg) || File.basename(arg) == 'pvrgpu_pco.c'
    next
  else
    flags << arg
  end
end
object = File.join(output, 'generate.o')
abort 'generator compilation failed' unless system(compiler, *flags, '-c',
  File.join(repo, 'tools/pco-fixtures/generate_multisample_texture.c'), '-o', object,
  chdir: entry.fetch('directory'))
ninja = File.read(File.join(build, 'build.ninja'))
link_line = ninja[/build src\/gallium\/drivers\/pvrgpu\/pvrgpu_pco_lowering_test:.*\n LINK_ARGS = ([^\n]+)/, 1]
abort 'Mesa PCO test link command unavailable' unless link_line
cpp_entry = database.find { |item| File.extname(item.fetch('file')) == '.cpp' }
cxx = (cpp_entry['arguments'] || Shellwords.split(cpp_entry.fetch('command'))).first
generator = File.join(output, 'generate')
abort 'generator linking failed' unless system(cxx, object, *Shellwords.split(link_line),
  '-o', generator, chdir: build)
21.times do |kind|
  abort "native fixture #{kind} generation failed" unless system(generator, kind.to_s,
    File.join(output, "texture-#{kind}.bin"), out: File.join(output, "texture-#{kind}.log"),
    err: [:child, :out])
end
test = File.join(output, 'pco-multisample-texture-test')
abort 'ISS test compilation failed' unless system(cxx, '-std=c++17', '-O1',
  '-I' + File.join(repo, 'src'), '-I' + File.join(repo, 'src/systemc'),
  File.join(repo, 'tests/pco_multisample_texture_test.cpp'),
  File.join(repo, 'src/systemc/shader/pco_iss.cpp'), '-o', test)
abort 'native fixture/ISS validation failed' unless system(test, output)
puts "Native multisample fixture/ISS artifacts: #{output}"
RUBY
