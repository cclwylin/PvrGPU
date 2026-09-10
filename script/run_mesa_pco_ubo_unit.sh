#!/usr/bin/env bash
# Build the current generic UBO compiler ABI regression, or the shared-budget
# matrix, against existing Mesa libraries. No installed driver or model binary
# is changed. Usage: run_mesa_pco_ubo_unit.sh [ubo|shared-budget]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    source "${REPO_DIR}/config/local.env"
    set +a
fi
MESA_TEST_BUILD="${PVRGPU_MESA_PVRGPU_BUILD_DIR:-${PVRGPU_BUILD_DIR:?Set PVRGPU_BUILD_DIR}/mesa-pvrgpu}"
MESA_TEST_OUT="$(mktemp -d "${TMPDIR:-/tmp}/pvrgpu-ubo-compiler.XXXXXX")"
PVRGPU_PCO_UNIT_MODE="${1:-ubo}"
if [[ "${PVRGPU_PCO_UNIT_MODE}" != "ubo" &&
      "${PVRGPU_PCO_UNIT_MODE}" != "shared-budget" ]]; then
    echo "usage: $0 [ubo|shared-budget]" >&2
    exit 2
fi
export REPO_DIR MESA_TEST_BUILD MESA_TEST_OUT PVRGPU_PCO_UNIT_MODE
ruby -rjson -rshellwords <<'RUBY'
build = File.expand_path(ENV.fetch('MESA_TEST_BUILD'))
repo = File.expand_path(ENV.fetch('REPO_DIR'))
output = File.expand_path(ENV.fetch('MESA_TEST_OUT'))
mode = ENV.fetch('PVRGPU_PCO_UNIT_MODE')
database = JSON.parse(File.read(File.join(build, 'compile_commands.json')))
entry = database.find { |item| File.basename(item.fetch('file')) == 'pvrgpu_pco.c' }
abort 'Mesa PCO compiler command missing' unless entry
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
objects = []
test_source = mode == 'shared-budget' ?
  'src/gallium/drivers/pvrgpu/tests/pvrgpu_pco_shared_budget_test.c' :
  'tests/pvrgpu_generic_uniform_buffer_test.c'
['src/gallium/drivers/pvrgpu/pvrgpu_pco.c',
 test_source].each do |source|
  object = File.join(output, File.basename(source) + '.o')
  abort 'C compilation failed' unless system(compiler,
    '-I' + File.join(repo, 'src/gallium/drivers/pvrgpu'), *flags, '-c', File.join(repo, source),
    '-o', object, chdir: entry.fetch('directory'))
  objects << object
end
ninja = File.read(File.join(build, 'build.ninja'))
link_line = ninja[/build src\/gallium\/drivers\/pvrgpu\/pvrgpu_pco_lowering_test:.*\n LINK_ARGS = ([^\n]+)/, 1]
abort 'Mesa PCO test link command unavailable' unless link_line
cpp_entry = database.find { |item| File.extname(item.fetch('file')) == '.cpp' }
cpp_command = cpp_entry['arguments'] || Shellwords.split(cpp_entry.fetch('command'))
executable = File.join(output, mode == 'shared-budget' ?
  'pvrgpu-pco-shared-budget-test' : 'generic-ubo-abi-test')
abort 'UBO compiler test linking failed' unless system(cpp_command.first,
  *objects, *Shellwords.split(link_line), '-o', executable, chdir: build)
abort "#{mode} compiler test failed" unless system(executable)
puts "#{mode} compiler unit artifacts: #{output}"
RUBY
