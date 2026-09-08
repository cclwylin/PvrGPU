# SPDX-License-Identifier: MIT
# Mechanical capture of real compiler output; no shader results are generated.
require 'digest'
abort 'usage: generate_compute_shared_fixtures.rb compiler-artifact-dir output.h [images]' unless (2..3).cover?(ARGV.size)
abort 'third argument must be images or textures (omit it for shared fixtures)' if ARGV.size == 3 && !%w[images textures].include?(ARGV[2])
images = ARGV[2] == 'images'
textures = ARGV[2] == 'textures'
first, last = textures ? [300, 303] : images ? [32, 38] : [24, 30]
stem = textures ? 'ComputeTexturePco' : images ? 'ComputeImagePco' : 'ComputeSharedPco'
origin = images ? 'actual compute compiler NIR tests' : 'pvrgpu_compute_compiler_test.c'
lines = ["// Generated from #{origin}, native gx6250 PCO.",
         '#pragma once', '#include "compute_types.h"', '#include <stdexcept>',
         'namespace pvrgpu::stub {',
         "inline const std::vector<std::uint8_t> &#{stem}Fixture(unsigned kind) {",
         '  static const std::vector<std::uint8_t> binaries[] = {']
abis = []
(first..last).each do |kind|
  bytes = File.binread(File.join(ARGV[0], "compute-#{kind}.bin"))
  abi = File.read(File.join(ARGV[0], "compute-#{kind}.abi"))
  abis << abi.scan(/([a-z_]+)=([^\s]+)/).to_h
  lines << "    // Kind #{kind}, SHA256 #{Digest::SHA256.hexdigest(bytes)}"
  abi.each_line { |line| lines << '    // ' + line.strip }
  lines << '    {'
  bytes.bytes.each_slice(12) { |chunk| lines << '      ' + chunk.map { |b| '0x%02x' % b }.join(', ') + ',' }
  lines << '    },'
end
lines += ['  };', "  if (kind < #{first} || kind > #{last}) throw std::runtime_error(\"#{images ? 'compute' : 'shared'} fixture kind\");",
          "  return binaries[kind - #{first}];", '}',
          "inline ComputePcoAbi #{stem}Abi(unsigned kind) {", '  ComputePcoAbi a;', '  switch (kind) {']
abis.each_with_index do |abi, index|
  lines << "  case #{index + first}:"
  %w[temps inputs coefficients shareds entry].zip(%w[temps vertex_inputs coefficients shareds entry_offset]).each do |from, to|
    lines << "    a.stage.#{to} = #{abi.fetch(from)};"
  end
  lines << "    a.local_size = {#{abi.fetch('local_size')}};"
  { 'local_index' => 'local_invocation_index', 'workgroup_id' => 'workgroup_id',
    'num_workgroups' => 'num_workgroups', 'ubo' => 'stage.uniform_buffer_descriptor',
    'ssbo' => 'storage_buffer_descriptor', 'push' => 'stage.push_constant' }.each do |from, to|
    start, count = abi.fetch(from).split(',')
    lines << "    a.#{to}_start = #{start}; a.#{to}_count = #{count};"
  end
  { 'ubo_used' => 'uniform_buffer_used_mask', 'ssbo_used' => 'storage_buffer_used_mask',
    'read' => 'storage_buffer_read_mask', 'write' => 'storage_buffer_write_mask' }.each do |from, to|
    lines << "    a.#{to} = #{abi.fetch(from)};"
  end
  bytes, start, count = abi.fetch('workgroup_shared').split(',')
  lines << "    a.shared_memory_bytes = #{bytes};"
  lines << "    a.shared_memory_descriptor_start = #{start}; a.shared_memory_descriptor_count = #{count};"
  if images
    start, count = abi.fetch('images').split(',')
    lines << "    a.image_descriptor_start = #{start}; a.image_descriptor_count = #{count};"
    %w[used read write].each do |mode|
      lines << "    a.image_#{mode}_mask = #{abi.fetch('image_' + mode)};"
    end
  end
  lines << '    a.sampled_texture_count = 1;' if textures
  lines << '    break;'
end
lines += ['  default: throw std::runtime_error("shared fixture ABI kind");', '  }', '  return a;', '}', '} // namespace pvrgpu::stub']
File.write(ARGV[1], lines.join("\n") + "\n")
