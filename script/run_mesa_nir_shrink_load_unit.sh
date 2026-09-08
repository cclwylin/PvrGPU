#!/usr/bin/env bash
# Exercise the configured real NIR shrink pass without rebuilding/installing Mesa.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    source "${REPO_DIR}/config/local.env"
    set +a
fi
MESA_TEST_BUILD="${PVRGPU_MESA_PVRGPU_BUILD_DIR:-${PVRGPU_BUILD_DIR:?Set PVRGPU_BUILD_DIR}/mesa-pvrgpu}"
MESA_TEST_OUT="$(mktemp -d "${PVRGPU_TMP_ROOT:-${TMPDIR:-/tmp}}/pvrgpu-nir-shrink-load.XXXXXX")"
export REPO_DIR MESA_TEST_BUILD MESA_TEST_OUT
ruby -rjson -rshellwords -ropen3 -rdigest <<'RUBY'
repo, build, output = ENV.values_at('REPO_DIR', 'MESA_TEST_BUILD', 'MESA_TEST_OUT').map { |p| File.expand_path(p) }
puts "NIR shrink-load ASan/UBSan artifacts: #{output}"
database = JSON.parse(File.read(File.join(build, 'compile_commands.json')))
entry = database.find { |item| File.basename(item.fetch('file')) == 'nir_opt_shrink_vectors.c' }
abort 'Mesa NIR compile command unavailable' unless entry
directory = entry.fetch('directory')
source = File.expand_path(ENV['PVRGPU_NIR_SHRINK_TEST_SOURCE'] || File.expand_path(entry.fetch('file'), directory))
abort "Mesa NIR source missing: #{source}" unless File.file?(source)
puts "NIR source under test: #{source}"
source_hash = Digest::SHA256.file(source).hexdigest
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
san = %w[-O1 -g -UNDEBUG -fsanitize=address,undefined -fno-omit-frame-pointer]
object = File.join(output, 'nir-shrink-load-test.o')
log = File.join(output, 'compile.log')
abort "Compile failed; see #{log}" unless system(compiler, *flags, *san,
  '-DPVRGPU_NIR_SHRINK_SOURCE="' + source + '"', '-c', File.join(repo, 'tests/mesa_nir_shrink_load_footprint_test.c'),
  '-o', object, chdir: directory, out: log, err: [:child, :out])
support_objects = %w[nir.c nir_validate.c].map do |name|
  support = database.find { |item| File.basename(item.fetch('file')) == name }
  abort "Mesa NIR support compile command unavailable: #{name}" unless support
  support_object = File.join(output, name + '.o')
  support_source = File.expand_path(support.fetch('file'), support.fetch('directory'))
  log = File.join(output, 'compile-' + name + '.log')
  abort "NIR support compile failed; see #{log}" unless system(compiler, *flags, *san,
    '-c', support_source, '-o', support_object, chdir: directory, out: log, err: [:child, :out])
  support_object
end
# Query only: never invoke a build or alter shared objects. The tested pass is
# renamed in our TU, so archives cannot silently select the stale implementation.
target = 'src/gallium/drivers/pvrgpu/pvrgpu_pco_geometry_test'
commands, status = Open3.capture2('ninja', '-C', build, '-t', 'commands', target)
abort 'Mesa native-test link command unavailable' unless status.success?
link = Shellwords.split(commands.lines.last || '')
old_object = target + '.p/tests_pvrgpu_pco_geometry_test.c.o'
abort 'Unexpected Mesa native-test link command' unless link.include?(old_object) && link.include?('-o')
binary = File.join(output, 'nir-shrink-load-test')
link[link.index('-o') + 1] = binary
link[link.index(old_object), 1] = [object, *support_objects]
link.concat(san)
link << (RUBY_PLATFORM.include?('darwin') ? '-Wl,-dead_strip' : '-Wl,--gc-sections')
log = File.join(output, 'link.log')
abort "Link failed; see #{log}" unless system(*link, chdir: build, out: log, err: [:child, :out])
env = {'ASAN_OPTIONS' => 'detect_leaks=0', 'UBSAN_OPTIONS' => 'halt_on_error=1'}
log = File.join(output, 'test.log')
ok = system(env, binary, out: log, err: [:child, :out])
File.write(File.join(output, 'receipt.json'), JSON.pretty_generate({
  schema: 'pvrgpu.nir-shrink-load-unit.v1', source: source, source_sha256: source_hash,
  source_unchanged: Digest::SHA256.file(source).hexdigest == source_hash,
  fixture: File.join(repo, 'tests/mesa_nir_shrink_load_footprint_test.c'),
  fixture_sha256: Digest::SHA256.file(File.join(repo, 'tests/mesa_nir_shrink_load_footprint_test.c')).hexdigest,
  binary: binary, binary_sha256: Digest::SHA256.file(binary).hexdigest,
  sanitizers: %w[address undefined], outcome: ok ? 'PASS' : 'FAIL'
}) + "\n")
puts File.read(log)
abort "NIR shrink-load verification failed; see #{log}" unless ok
RUBY
