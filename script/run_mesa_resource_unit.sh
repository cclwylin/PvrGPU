#!/usr/bin/env bash
# Build focused resource/clear tests with an existing Mesa build's actual
# compiler configuration. Generated executables stay outside the source tree.
# Usage: bash script/run_mesa_resource_unit.sh [all|blit|clear|ubo|push-map|compute|surface|vertex|command|texture|boundary|flush|depth-upload|payload-budget|native-present|packed-load|packed-store|packed-descriptor]
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
abort 'usage: run_mesa_resource_unit.sh [all|blit|clear|ubo|push-map|compute|surface|vertex|command|texture|boundary|flush|depth-upload|payload-budget|native-present|packed-load|packed-store|packed-descriptor]' unless
  %w[all blit clear ubo push-map compute surface vertex command texture boundary flush depth-upload payload-budget native-present packed-load packed-store packed-descriptor].include?(selected)
tests = { 'blit' => ['pvrgpu_resource.c', 'pvrgpu_msaa_blit_test.c'],
          'clear' => ['pvrgpu_clear.c', 'pvrgpu_clear_storage_test.c'],
          'ubo' => ['pvrgpu_context.c', 'pvrgpu_uniform_buffer_snapshot_test.c'],
          'push-map' => ['pvrgpu_context.c', 'pvrgpu_push_constant_map_test.c'],
          'compute' => ['pvrgpu_context.c', 'pvrgpu_compute_snapshot_test.c'],
          'surface' => ['pvrgpu_resource.c', 'pvrgpu_surface_span_test.c'],
          'boundary' => ['pvrgpu_resource.c', 'pvrgpu_framebuffer_boundary_test.c'],
          'flush' => ['pvrgpu_context.c', 'pvrgpu_flush_test.c'],
          'native-present' => ['pvrgpu_context.c', 'pvrgpu_native_present_guard_test.c'],
          'packed-load' => ['pvrgpu_context.c', 'pvrgpu_packed_color_load_test.c'],
          'packed-store' => ['pvrgpu_resource.c', 'pvrgpu_packed_color_store_test.c'],
          'packed-descriptor' => ['pvrgpu_pco.c', 'pvrgpu_packed_color_descriptor_test.c'],
          'depth-upload' => ['pvrgpu_resource.c', 'pvrgpu_depth_upload_copy_test.c'],
          'payload-budget' => ['pvrgpu_context.c', 'pvrgpu_payload_budget_test.c'],
          'vertex' => ['pvrgpu_context.c', 'pvrgpu_vertex_attribute_fetch_test.c'],
          'texture' => ['pvrgpu_context.c', 'pvrgpu_texture_view_snapshot_test.c'],
          'command' => ['pvrgpu_cmd.c', 'pvrgpu_command_defaults_test.c'] }
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
  if name == 'texture'
    # Snapshot admission calls the actual compiler's read-only sampler-use
    # proof. Compile the current implementation privately rather than linking
    # a possibly stale installed driver or replacing the proof with a stub.
    pco_entry = database.find { |item| File.basename(item.fetch('file')) == 'pvrgpu_pco.c' }
    abort 'Mesa compile command missing for pvrgpu_pco.c' unless pco_entry
    pco_command = pco_entry['arguments']&.dup || Shellwords.split(pco_entry.fetch('command'))
    pco_args = []
    until pco_command.empty?
      argument = pco_command.shift
      if %w[-o -MF -MQ -MT].include?(argument)
        pco_command.shift
      elsif %w[-MD -MMD].include?(argument)
        next
      elsif File.basename(argument) == 'pvrgpu_pco.c'
        pco_args << File.join(repo, 'src/gallium/drivers/pvrgpu/pvrgpu_pco.c')
      else
        pco_args << argument
      end
    end
    pco_object = File.join(output, 'pvrgpu_pco.o')
    pco_args.concat(['-ffunction-sections', '-fdata-sections', '-o', pco_object])
    abort 'texture compiler proof compilation failed' unless
      system(*pco_args, chdir: pco_entry.fetch('directory'))
    args.concat([pco_object, 'src/compiler/nir/libnir.a',
                 'src/compiler/libcompiler.a',
                 RUBY_PLATFORM.include?('darwin') ? '-lc++' : '-lstdc++'])
  end
  if name == 'depth-upload'
    pack = database.find { |item| item.fetch('file').end_with?('/main/pack.c') }
    abort 'Mesa compile command missing for main/pack.c' unless pack
    mesa = File.dirname(File.dirname(File.expand_path(pack.fetch('file'), pack.fetch('directory'))))
    args.concat(["-I#{mesa}", 'src/mesa/libmesa.a',
                 'src/mesa/glapi/shared-glapi/libglapi.a'])
  end
  args.concat(['src/util/libmesa_util.a', 'src/c11/impl/libmesa_util_c11.a',
               '-lm', '-lpthread', '-o', executable])
  puts "Building #{name} with #{driver}'s Mesa compiler flags"
  abort "#{name} compilation failed" unless system(*args, chdir: entry.fetch('directory'))
  abort "#{name} tests failed" unless system(executable)
end
puts "Mesa unit artifacts: #{output}"
RUBY
