#!/usr/bin/env bash
# Regenerate real uniform-branch FS binaries and test native SMP continuations.
# All outputs are private. Does not install or rebuild a shared Mesa/SystemC.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    source "${REPO_DIR}/config/local.env"
    set +a
fi
MESA_TEST_BUILD="${PVRGPU_MESA_PVRGPU_BUILD_DIR:-${PVRGPU_BUILD_DIR:?}/mesa-pvrgpu}"
MESA_TEST_OUT="$(mktemp -d "${PVRGPU_TMP_ROOT:-${TMPDIR:-/tmp}}/pvrgpu-fragment-select.XXXXXX")"
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
  elsif %w[-c -MD -MMD].include?(arg) || File.basename(arg) == 'pvrgpu_pco.c'
    next
  else
    flags << arg
  end
end
san = %w[-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer]
objects = %w[src/gallium/drivers/pvrgpu/pvrgpu_pco.c tests/pvrgpu_fragment_texture_select_test.c].map.with_index do |source, index|
  object = File.join(output, "compiler-#{index}.o")
  abort "compile #{source} failed" unless system(compiler, *flags, *san, '-c', File.join(repo, source), '-o', object,
    chdir: entry.fetch('directory'), out: File.join(output, "compile-#{index}.log"), err: [:child, :out])
  object
end
target = 'src/gallium/drivers/pvrgpu/pvrgpu_pco_geometry_test'
line = IO.popen(['ninja', '-C', build, '-t', 'commands', target], &:read).lines.last
args = Shellwords.split(line)
args[args.index('-o') + 1] = File.join(output, 'compiler-test')
args[args.index(target + '.p/tests_pvrgpu_pco_geometry_test.c.o')] = objects[1]
args.insert(args.index('src/gallium/drivers/pvrgpu/libpvrgpu.a'), objects[0])
args.concat(san)
abort 'native compiler test link failed' unless system(*args, chdir: build,
  out: File.join(output, 'link.log'), err: [:child, :out])
fixtures = %w[fragment.bin fragment-poison.bin upper.bin].map { |name| File.join(output, name) }
env = {'ASAN_OPTIONS' => 'detect_leaks=0', 'UBSAN_OPTIONS' => 'halt_on_error=1'}
abort 'compiler/select guard test failed' unless system(env, File.join(output, 'compiler-test'), *fixtures,
  out: File.join(output, 'compiler-test.log'), err: [:child, :out])
cxx = args.first
test = File.join(output, 'native-test')
abort 'native ISS test compilation failed' unless system(cxx, '-std=c++17', *san,
  '-I' + File.join(repo, 'src'), '-I' + File.join(repo, 'src/systemc'),
  File.join(repo, 'tests/pco_fragment_texture_select_test.cpp'),
  File.join(repo, 'src/systemc/shader/pco_iss.cpp'), '-o', test,
  out: File.join(output, 'native-compile.log'), err: [:child, :out])
abort 'native select/continuation test failed' unless system(env, test, *fixtures,
  out: File.join(output, 'native-test.log'), err: [:child, :out])
puts File.read(File.join(output, 'compiler-test.log'))
puts File.read(File.join(output, 'native-test.log'))
puts "Native fragment select ASan/UBSan artifacts: #{output}"
RUBY
