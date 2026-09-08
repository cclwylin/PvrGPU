#!/usr/bin/env bash
# Build focused resource/clear tests with an existing Mesa build's actual
# compiler configuration. Generated executables stay outside the source tree.
# Usage: bash script/run_mesa_resource_unit.sh [all|blit|clear|ubo|compute|surface|vertex]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${REPO_DIR}/config/local.env" ]]; then
    set -a
    source "${REPO_DIR}/config/local.env"
    set +a
fi
MESA_TEST_BUILD="${PVRGPU_MESA_PVRGPU_BUILD_DIR:-${PVRGPU_BUILD_DIR:?Set PVRGPU_BUILD_DIR or PVRGPU_MESA_PVRGPU_BUILD_DIR}/mesa-pvrgpu}"
MESA_TEST_TMP="${PVRGPU_TMP_ROOT:-${TMPDIR:-/tmp}}"
MESA_TEST_OUT="$(mktemp -d "${MESA_TEST_TMP%/}/pvrgpu-mesa-unit.XXXXXX")"
export MESA_TEST_BUILD MESA_TEST_OUT REPO_DIR

ruby -rjson -rshellwords - "${1:-all}" <<'RUBY'
build = File.expand_path(ENV.fetch('MESA_TEST_BUILD'))
repo = File.expand_path(ENV.fetch('REPO_DIR'))
output = File.expand_path(ENV.fetch('MESA_TEST_OUT'))
abort 'Mesa test build/output must be outside the source tree' if
  [build, output].any? { |path| path == repo || path.start_with?(repo + '/') }
database = JSON.parse(File.read(File.join(build, 'compile_commands.json')))
selected = ARGV.fetch(0)
abort 'usage: run_mesa_resource_unit.sh [all|blit|clear|ubo|compute|surface|vertex]' unless
  %w[all blit clear ubo compute surface vertex].include?(selected)
tests = { 'blit' => ['pvrgpu_resource.c', 'pvrgpu_msaa_blit_test.c'],
          'clear' => ['pvrgpu_clear.c', 'pvrgpu_clear_storage_test.c'],
          'ubo' => ['pvrgpu_context.c', 'pvrgpu_uniform_buffer_snapshot_test.c'],
          'compute' => ['pvrgpu_context.c', 'pvrgpu_compute_snapshot_test.c'],
          'surface' => ['pvrgpu_resource.c', 'pvrgpu_surface_span_test.c'],
          'vertex' => ['pvrgpu_context.c', 'pvrgpu_vertex_attribute_fetch_test.c'] }
tests.each do |name, (driver, source)|
  next unless selected == 'all' || selected == name
  entry = database.find { |item| File.basename(item.fetch('file')) == driver }
  abort "Mesa compile command missing for #{driver}" unless entry
  command = entry['arguments'] || Shellwords.split(entry.fetch('command'))
  args = []
  until command.empty?
    argument = command.shift
    if %w[-o -MF -MQ -MT].include?(argument)
      command.shift
    elsif %w[-c -MD -MMD].include?(argument)
      next
    elsif File.basename(argument) == driver
      args << File.join(repo, 'tests', source)
    else
      args << argument
    end
  end
  executable = File.join(output, "pvrgpu-#{name}-test")
  if RUBY_PLATFORM.include?('darwin')
    args << '-Wl,-dead_strip'
  else
    args.concat(%w[-ffunction-sections -fdata-sections -Wl,--gc-sections])
  end
  args.concat(['src/util/libmesa_util.a', 'src/c11/impl/libmesa_util_c11.a',
               '-lm', '-lpthread', '-o', executable])
  puts "Building #{name} with #{driver}'s Mesa compiler flags"
  abort "#{name} compilation failed" unless system(*args, chdir: entry.fetch('directory'))
  abort "#{name} tests failed" unless system(executable)
end
puts "Mesa unit artifacts: #{output}"
RUBY
