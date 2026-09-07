#!/usr/bin/env bash
# Isolated compute compiler tests/fixtures; never replaces a Mesa/model runtime.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    source "${REPO_DIR}/config/local.env"
    set +a
fi
MESA_TEST_BUILD="${PVRGPU_MESA_PVRGPU_BUILD_DIR:-${PVRGPU_BUILD_DIR:?Set PVRGPU_BUILD_DIR}/mesa-pvrgpu}"
MESA_TEST_OUT="$(mktemp -d "${PVRGPU_WORK_ROOT:?Set PVRGPU_WORK_ROOT}/tmp/pvrgpu-compute-compiler.XXXXXX")"
export REPO_DIR MESA_TEST_BUILD MESA_TEST_OUT
ruby -rjson -rshellwords <<'RUBY'
build = File.expand_path(ENV.fetch('MESA_TEST_BUILD'))
repo = File.expand_path(ENV.fetch('REPO_DIR'))
output = File.expand_path(ENV.fetch('MESA_TEST_OUT'))
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
['src/gallium/drivers/pvrgpu/pvrgpu_pco.c', 'tests/pvrgpu_compute_compiler_test.c'].each do |source|
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
executable = File.join(output, 'compute-compiler-test')
abort 'Compute compiler test linking failed' unless system(cpp_command.first,
  *objects, *Shellwords.split(link_line), '-o', executable, chdir: build)
abort 'Compute compiler test failed' unless system(executable, output)
puts "Compute compiler unit artifacts: #{output}"
RUBY
