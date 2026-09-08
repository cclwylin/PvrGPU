#!/usr/bin/env bash
# Private true-Mesa producer and native texture BIAS verification. No install.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    source "${REPO_DIR}/config/local.env"
    set +a
fi
MESA_TEST_BUILD="${PVRGPU_MESA_PVRGPU_BUILD_DIR:-${PVRGPU_BUILD_DIR:?}/mesa-pvrgpu}"
MESA_TEST_OUT="$(mktemp -d "${PVRGPU_TMP_ROOT:-${TMPDIR:-/tmp}}/pvrgpu-texture-bias.XXXXXX")"
export REPO_DIR MESA_TEST_BUILD MESA_TEST_OUT
ruby -rjson -rshellwords -rdigest <<'RUBY' 2>&1 | tee "${MESA_TEST_OUT}/test.log"
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
receipt = {'schema'=>'pvrgpu.texture-bias-unit.v1', 'status'=>'RUNNING',
  'repo'=>repo, 'mesa_build'=>build, 'compiler_template_source'=>entry.fetch('file'),
  'actual_driver_source'=>File.join(repo, 'src/gallium/drivers/pvrgpu/pvrgpu_pco.c'),
  'repo_driver_include_first'=>true, 'runtime_installed'=>false, 'source_sha256'=>{}}
%w[src/gallium/drivers/pvrgpu/pvrgpu_pco.c tests/pvrgpu_texture_bias_test.c
   tests/texture_bias_test.cpp tests/pco_texture_bias_fixtures.h
   src/systemc/shader/pco_iss.h src/systemc/shader/pco_iss.cpp
   src/systemc/shader/usc_cluster.cpp src/systemc/common/functional_types.h
   src/systemc/texture/texture_unit.cpp src/systemc/texture/texture_filter.cpp].each do |source|
  receipt['source_sha256'][source]=Digest::SHA256.file(File.join(repo,source)).hexdigest
end
receipt_path=File.join(output,'receipt.json')
File.write(receipt_path,JSON.pretty_generate(receipt)+"\n")
objects = %w[src/gallium/drivers/pvrgpu/pvrgpu_pco.c tests/pvrgpu_texture_bias_test.c].map do |source|
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
abort 'native bias compilation failed' unless system(compiler_test, output)
receipt['native_pco_sha256']=(0...6).map { |kind|
  Digest::SHA256.file(File.join(output,"bias-#{kind}.pco")).hexdigest }
puts "True Mesa BIAS artifacts: #{output}"
unless ENV['PVRGPU_TEXTURE_BIAS_COMPILE_ONLY'] == '1'
  sources = %w[tests/texture_bias_test.cpp model_stub/memory_pool.cpp model_stub/model_types.cpp
    src/systemc/common/functional_types.cpp src/systemc/common/pipeline_state.cpp
    src/systemc/shader/pco_iss.cpp src/systemc/shader/usc_cluster.cpp
    src/systemc/texture/astc_decoder.cpp src/systemc/texture/texture_filter.cpp src/systemc/texture/texture_unit.cpp
    src/systemc/cache_mmu/cache_array.cpp src/systemc/memory/dram_address_space.cpp
    src/systemc/memory/gpu_memory_system.cpp]
  includes = %w[model_stub src src/systemc].map { |s| '-I' + File.join(repo, s) }
  systemc = Shellwords.split(IO.popen(['pkg-config', '--cflags', '--libs', 'systemc'], &:read))
  native = File.join(output, 'texture-bias-test')
  abort 'native BIAS compile failed' unless system(cxx, '-std=c++17', '-O1', '-g',
    '-fsanitize=address,undefined,float-cast-overflow', '-fno-omit-frame-pointer',
    *includes, *sources.map { |s| File.join(repo, s) }, *systemc, '-o', native)
  abort 'native BIAS verification failed' unless system({'ASAN_OPTIONS'=>'detect_leaks=0',
    'UBSAN_OPTIONS'=>'halt_on_error=1'}, native, output)
  abort 'checked-in BIAS fixture verification failed' unless system({'ASAN_OPTIONS'=>'detect_leaks=0',
    'UBSAN_OPTIONS'=>'halt_on_error=1'}, native)
  receipt['native_test_sha256']=Digest::SHA256.file(native).hexdigest
  puts "True Mesa BIAS native ASan/UBSan verified: #{output}"
end
receipt['status']=ENV['PVRGPU_TEXTURE_BIAS_COMPILE_ONLY']=='1' ? 'COMPILE_ONLY' : 'PASS'
File.write(receipt_path,JSON.pretty_generate(receipt)+"\n")
RUBY
