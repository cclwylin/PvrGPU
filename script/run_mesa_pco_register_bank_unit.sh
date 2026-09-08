#!/usr/bin/env bash
# Test physical PCO parallel copies with ASan/UBSan in an isolated directory.
# Uses the configured Mesa source/build without changing an installed runtime.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    source "${REPO_DIR}/config/local.env"
    set +a
fi
MESA_TEST_BUILD="${PVRGPU_MESA_PVRGPU_BUILD_DIR:-${PVRGPU_BUILD_DIR:?Set PVRGPU_BUILD_DIR}/mesa-pvrgpu}"
MESA_TEST_OUT="$(mktemp -d "${PVRGPU_TMP_ROOT:-${TMPDIR:-/tmp}}/pvrgpu-register-bank.XXXXXX")"
export REPO_DIR MESA_TEST_BUILD MESA_TEST_OUT
ruby -rjson -rshellwords -ropen3 <<'RUBY'
repo, build, output = ENV.values_at('REPO_DIR', 'MESA_TEST_BUILD', 'MESA_TEST_OUT').map { |p| File.expand_path(p) }
puts "PCO register-bank ASan/UBSan artifacts: #{output}"
database = JSON.parse(File.read(File.join(build, 'compile_commands.json')))
entry = database.find { |item| File.basename(item.fetch('file')) == 'pco_ra.c' }
abort 'Mesa PCO allocator compile command unavailable' unless entry
directory = entry.fetch('directory')
source = File.expand_path(ENV['PVRGPU_PCO_RA_TEST_SOURCE'] || File.expand_path(entry.fetch('file'), directory))
abort "Mesa allocator source missing: #{source}" unless File.file?(source)
puts "Allocator under test: #{source}"
command = entry['arguments'] || Shellwords.split(entry.fetch('command'))
compiler = command.shift
flags = []
until command.empty?
  arg = command.shift
  if %w[-o -MF -MQ -MT].include?(arg)
    command.shift
  elsif %w[-c -MD -MMD -DNDEBUG].include?(arg) || arg == entry.fetch('file')
    next
  else
    flags << arg
  end
end
flags << '-UNDEBUG'
san = %w[-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer]
sources = [File.join(repo, 'tests/pco_register_bank_copies_test.c')]
# Recompile IR allocation and generated opcode metadata together. An existing
# archive may predate regenerated opcode headers in an actively developed tree.
%w[pco.c pco_info.c].each do |name|
  support = database.find { |item| File.basename(item.fetch('file')) == name }
  abort "Mesa supporting compile command missing: #{name}" unless support
  sources << File.expand_path(support.fetch('file'), support.fetch('directory'))
end
objects = sources.map.with_index do |input, index|
  object = File.join(output, "test-#{index}.o")
  log = File.join(output, "compile-#{index}.log")
  abort "Compilation failed; see #{log}" unless system(compiler, *flags, *san,
    '-DPVRGPU_PCO_RA_SOURCE="' + source + '"', '-c', input, '-o', object,
    chdir: directory, out: log, err: [:child, :out])
  object
end
target = 'src/gallium/drivers/pvrgpu/pvrgpu_pco_geometry_test'
commands, status = Open3.capture2('ninja', '-C', build, '-t', 'commands', target)
abort 'Mesa PCO test link command unavailable' unless status.success?
link = Shellwords.split(commands.lines.last || '')
old_object = target + '.p/tests_pvrgpu_pco_geometry_test.c.o'
abort 'Unexpected Mesa PCO test link command' unless link.include?(old_object) && link.include?('-o')
binary = File.join(output, 'register-bank-test')
link[link.index('-o') + 1] = binary
link[link.index(old_object), 1] = objects
link.concat(san)
link << (RUBY_PLATFORM.include?('darwin') ? '-Wl,-dead_strip' : '-Wl,--gc-sections')
log = File.join(output, 'link.log')
abort "Link failed; see #{log}" unless system(*link, chdir: build, out: log, err: [:child, :out])
env = {'ASAN_OPTIONS' => 'detect_leaks=0', 'UBSAN_OPTIONS' => 'halt_on_error=1'}
log = File.join(output, 'test.log')
ok = system(env, binary, out: log, err: [:child, :out])
puts File.read(log)
abort "PCO parallel-copy verification failed; see #{log}" unless ok
RUBY
