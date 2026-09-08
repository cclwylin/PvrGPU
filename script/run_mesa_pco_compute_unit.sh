#!/usr/bin/env bash
# Isolated compute compiler tests/fixtures; never replaces a Mesa/model runtime.
set -euo pipefail
if [[ $# -gt 1 ]]; then
    echo "usage: $0 [all|shared|images]" >&2
    exit 64
fi
MESA_TEST_SELECTION="${1:-all}"
case "${MESA_TEST_SELECTION}" in
    all|shared|images) ;;
    -h|--help)
        echo "usage: $0 [all|shared|images]"
        echo 'all (default): buffer/shared 0..30 and image 32..38 compiler tests.'
        echo 'shared: existing buffer tests plus shared fixtures 24..30.'
        echo 'images: image fixtures 32..38 and invalid-image compiler tests.'
        echo 'Native binaries, ABI files and generated headers stay in a fresh private directory.'
        exit 0 ;;
    *) echo "invalid compute compiler selection: ${MESA_TEST_SELECTION}" >&2; exit 64 ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    source "${REPO_DIR}/config/local.env"
    set +a
fi
MESA_TEST_BUILD="${PVRGPU_MESA_PVRGPU_BUILD_DIR:-${PVRGPU_BUILD_DIR:?Set PVRGPU_BUILD_DIR}/mesa-pvrgpu}"
MESA_TEST_OUT="$(mktemp -d "${PVRGPU_WORK_ROOT:?Set PVRGPU_WORK_ROOT}/tmp/pvrgpu-compute-compiler.XXXXXX")"
export REPO_DIR MESA_TEST_BUILD MESA_TEST_OUT MESA_TEST_SELECTION
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
compile_source = lambda do |source|
  object = File.join(output, File.basename(source) + '.o')
  abort 'C compilation failed' unless system(compiler,
    '-I' + File.join(repo, 'src/gallium/drivers/pvrgpu'), *flags, '-c', File.join(repo, source),
    '-o', object, chdir: entry.fetch('directory'))
  object
end
pco_object = compile_source.call('src/gallium/drivers/pvrgpu/pvrgpu_pco.c')
ninja = File.read(File.join(build, 'build.ninja'))
link_line = ninja[/build src\/gallium\/drivers\/pvrgpu\/pvrgpu_pco_lowering_test:.*\n LINK_ARGS = ([^\n]+)/, 1]
abort 'Mesa PCO test link command unavailable' unless link_line
cpp_entry = database.find { |item| File.extname(item.fetch('file')) == '.cpp' }
abort 'Mesa C++ linker command missing' unless cpp_entry
cpp_command = cpp_entry['arguments'] || Shellwords.split(cpp_entry.fetch('command'))
selection = ENV.fetch('MESA_TEST_SELECTION')
tests = []
tests << ['shared', 'tests/pvrgpu_compute_compiler_test.c', 'compute-compiler-test'] if selection != 'images'
tests << ['images', 'tests/pvrgpu_compute_image_compiler_test.c', 'compute-image-compiler-test'] if selection != 'shared'
tests.each do |kind, source, name|
  object = compile_source.call(source)
  executable = File.join(output, name)
  abort "#{kind} compiler test linking failed" unless system(cpp_command.first,
    pco_object, object, *Shellwords.split(link_line), '-o', executable, chdir: build)
  abort "#{kind} compiler test failed" unless system(executable, output)
  header = File.join(output, "pco_compute_#{kind == 'images' ? 'image' : 'shared'}_fixtures.h")
  generator = [File.join(repo, 'script/generate_compute_shared_fixtures.rb'), output, header]
  generator << 'images' if kind == 'images'
  abort "#{kind} fixture header generation failed" unless system('ruby', *generator)
  puts "Generated native #{kind} fixture header: #{header}"
end
puts "Compute compiler unit artifacts: #{output}"
RUBY
