/* SPDX-License-Identifier: MIT */
#include "shader/compute_iss.h"
#include "pco_compute_fixtures.h"

#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

using namespace pvrgpu::stub;
namespace {
void Check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template <class Fn> void Reject(Fn fn) {
  try { fn(); } catch (const std::exception &) { return; }
  throw std::runtime_error("expected fail-closed ADD64_32 validation");
}

// 未修改的原生指令群，擷取自 Mesa da14d65e 的 PCO_DEBUG_PRINT=cs,binary：
// dEQP-GLES31.functional.atomic_counter.get_inc_branch.
// 8_counters_100_calls_1_thread。group35/36 位於 0x184/0x192，
// group55/56 位於 0x260/0x26e；前者 r21/vi0，後者 vi0/vi1。
// 每對指令的第二群讀回第一群的兩個結果，再加 sh3，驗證 bank 與寫入標記。
// pco_isa.py F_REGBANK: TEMP=1, VTXIN=2；I_TWO_3B8I_3B8I 獨立編碼 db0/db1。
const std::array<std::vector<std::uint8_t>,2> kNativeGroups{{
  {0x37,0x86,0x00,0xe8,0x80,0x75,0x80,0x54,0x08,0x00,0xc0,0x95,0x80,0x20,
   0x37,0x86,0x00,0xe8,0xd5,0x40,0x88,0x03,0x80,0x00,0xc0,0x80,0x41,0xff},
  {0x37,0x86,0x00,0xe8,0x80,0x41,0x88,0x80,0x08,0x00,0xc0,0x00,0x81,0x22,
   0x37,0x86,0x00,0xe8,0x80,0x41,0x98,0x03,0x80,0x00,0xc0,0x80,0x41,0xff}
}};

std::vector<std::uint8_t> NativeProgram(unsigned kind) {
  auto bytes = kNativeGroups.at(kind);
  const auto &end = ComputePcoFixture(0); // 原生 NOP.END，不替代運算。
  bytes.insert(bytes.end(), end.begin(), end.end());
  return bytes;
}

void TestNativePair(unsigned kind) {
  const auto binary = NativeProgram(kind);
  const auto program = DecodeComputePcoProgram(binary);
  const auto &add = program.instructions.at(0);
  Check(add.opcode == PcoOpcode::kIntegerAdd64_32 &&
        add.target == (kind ? PcoWriteTarget::kVertexInput : PcoWriteTarget::kTemporary) &&
        add.output_index == (kind ? 0 : 21) &&
        add.output_target1 == PcoWriteTarget::kVertexInput &&
        add.output_index1 == kind && add.address_offset_signed,
        "native ADD64_32 lost its independent destination banks");
  Reject([&] { DecodePcoProgram(ShaderStage::kVertex, binary); });
  Reject([&] { DecodePcoProgram(ShaderStage::kFragment, binary); });
  ComputePcoAbi abi;
  abi.local_size = {32,1,1};
  abi.stage.temps = 22;
  abi.stage.vertex_inputs = 2;
  abi.stage.shareds = abi.stage.push_constant_count = 8;
  ValidateComputeProgram(program, abi);

  for (const auto low : {UINT32_C(0), UINT32_MAX, UINT32_C(0x80000000)}) {
    std::vector<std::uint32_t> shared(8);
    shared[0] = low; shared[3] = 7;
    auto task = MakeComputeTask(abi, shared, {1,1,1}, {0,0,0}, 0, 32);
    std::array<std::uint64_t,32> expected{};
    for (unsigned lane = 0; lane < 32; ++lane) {
      const auto high = lane % 2 ? UINT32_MAX : UINT32_C(0x10000);
      const auto offset = lane % 3 == 0 ? UINT32_MAX :
                          lane % 3 == 1 ? UINT32_C(1) : UINT32_C(0x80000000);
      expected[lane] = (std::uint64_t(high) << 32 | low) +
          static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(offset)));
      auto &state = task.lanes[lane];
      if (kind) {
        state.inputs[0] = offset; state.inputs[1] = high;
        state.inputs_written = 3;
      } else {
        state.temporaries[20] = offset; state.temporaries[21] = high;
        state.temporary_written.set(20); state.temporary_written.set(21);
      }
    }
    ComputeWorkgroupResult result;
    StepComputeTask(program, abi, task, {}, result);
    for (unsigned lane = 0; lane < 32; ++lane) {
      const auto &state = task.lanes[lane];
      Check((kind ? state.inputs[0] : state.temporaries[21]) ==
                static_cast<std::uint32_t>(expected[lane]) &&
            state.inputs[kind] == static_cast<std::uint32_t>(expected[lane] >> 32),
            "native ADD64_32 carry/borrow or aliased input read was corrupted");
    }
    StepComputeTask(program, abi, task, {}, result);
    for (unsigned lane = 0; lane < 32; ++lane) {
      const auto &state = task.lanes[lane];
      const auto value = expected[lane] + 7;
      Check(state.temporaries[0] == static_cast<std::uint32_t>(value) &&
            state.temporaries[1] == static_cast<std::uint32_t>(value >> 32),
            "following native ADD64_32 did not read both written result banks");
    }
    Check(result.stats.alu_instructions == 64 &&
          result.stats.memory_instructions == 0,
          "ADD64_32 pair changed instruction counts or invented memory traffic");
    StepComputeTask(program, abi, task, {}, result);
    Check(task.ended && result.stats.alu_instructions == 96,
          "native NOP.END did not finish all task lanes");
  }

  // 目的地逐一依各自 ABI span 驗證；TEMP 容量不能替 VTXIN 越界背書。
  for (unsigned destination = 0; destination < 2; ++destination) {
    auto bad = program;
    auto &i = bad.instructions[0];
    auto &target = destination ? i.output_target1 : i.target;
    auto &index = destination ? i.output_index1 : i.output_index;
    index = target == PcoWriteTarget::kTemporary ? abi.stage.temps : abi.stage.vertex_inputs;
    Reject([&] { ValidateComputeProgram(bad, abi); });
    --index;
    ValidateComputeProgram(bad, abi);
    target = PcoWriteTarget::kPixelOutput;
    Reject([&] { ValidateComputeProgram(bad, abi); });
    target = PcoWriteTarget::kNone;
    Reject([&] { ValidateComputeProgram(bad, abi); });
  }
  auto bad = program;
  bad.instructions[0].repeat_count = 2;
  Reject([&] { ValidateComputeProgram(bad, abi); });

  // 不支援的實體 bank、VTXIN64 及 TEMP256 不能被解碼器默默接受。
  for (unsigned bank : {0U,3U,4U,5U,6U,7U}) {
    auto corrupt = binary;
    corrupt[12] = static_cast<std::uint8_t>((corrupt[12] & ~0x40U) | ((bank & 1U) << 6));
    corrupt[13] = static_cast<std::uint8_t>((corrupt[13] & ~0x60U) | ((bank >> 1) << 5));
    Reject([&] { DecodeComputePcoProgram(corrupt); });
  }
  auto corrupt = binary;
  corrupt[13] |= 0x08; // 高目的 VTXIN index 的 bit6。
  Reject([&] { DecodeComputePcoProgram(corrupt); });
  corrupt = binary;
  corrupt[12] = 0xbf; // VTXIN63：db1 低位為 0，index 低六位全 1。
  corrupt[13] &= ~0x18U;
  Check(DecodeComputePcoProgram(corrupt).instructions[0].output_index1 == 63,
        "last physical VTXIN destination was rejected");
  corrupt[12] = 0xff; // TEMP255：db1=1，index 高兩位全 1。
  corrupt[13] = static_cast<std::uint8_t>((corrupt[13] & ~0x60U) | 0x18U);
  Check(DecodeComputePcoProgram(corrupt).instructions[0].output_index1 == 255,
        "last physical TEMP destination was rejected");
  corrupt[0] = 0x38; // 11-bit 目的地需四個 bytes，加 padding 後群長 16。
  corrupt[12] = 0xc0;
  corrupt[13] = static_cast<std::uint8_t>((corrupt[13] & ~0x78U) | 0x80U);
  corrupt.insert(corrupt.begin() + 14, {0x08,0xff}); // high d1[10:8]=1。
  Reject([&] { DecodeComputePcoProgram(corrupt); });
}

void TestNativeImmediateInput() {
  // 同一 case 的 get_dec_branch 版本，group52 位於 0x23e：
  // bbyp0bm_imm32 3200u -> vi0，後接原生 IMADD32 vi0=vi0+r20
  // 及 MBYP vi1=sh1。只擷取原生群，不改 shader allocator。
  std::vector<std::uint8_t> binary{
      0x86,0x92,0x40,0x13,0x80,0x0c,0x00,0x00,0x00,0x00,0x80,0x04,
      0x36,0x82,0x00,0xe2,0x80,0x41,0x10,0x54,0x00,0x00,0x80,0x04,
      0x36,0x82,0x00,0x87,0x81,0x08,0x00,0x00,0x00,0x81,0x04,0xff};
  const auto &end = ComputePcoFixture(0);
  binary.insert(binary.end(), end.begin(), end.end());
  const auto program = DecodeComputePcoProgram(binary);
  const auto &move = program.instructions[0];
  Check(move.opcode == PcoOpcode::kMoveImmediate &&
        move.target == PcoWriteTarget::kVertexInput && !move.output_index &&
        move.immediate == 3200, "native immediate lost its VTXIN destination");
  ComputePcoAbi abi;
  abi.local_size = {32,1,1}; abi.stage.vertex_inputs = 2;
  abi.stage.temps = 21;
  abi.stage.shareds = abi.stage.push_constant_count = 8;
  ValidateComputeProgram(program, abi);
  std::vector<std::uint32_t> shared(8);
  shared[1] = 0x10000;
  auto task = MakeComputeTask(abi, shared, {1,1,1}, {0,0,0}, 0, 32);
  for (unsigned lane = 0; lane < 32; ++lane) {
    task.lanes[lane].temporaries[20] = lane;
    task.lanes[lane].temporary_written.set(20);
  }
  ComputeWorkgroupResult result;
  StepComputeTask(program, abi, task, {}, result);
  for (const auto &lane : task.lanes)
    Check(lane.inputs[0] == 3200 && lane.inputs_written == 1,
          "native immediate failed to write or mark VTXIN");
  StepComputeTask(program, abi, task, {}, result);
  StepComputeTask(program, abi, task, {}, result);
  for (unsigned lane = 0; lane < 32; ++lane)
    Check(task.lanes[lane].inputs[0] == 3200 + lane &&
          task.lanes[lane].inputs[1] == shared[1] &&
          task.lanes[lane].inputs_written == 3,
          "native IMADD32/MBYP failed to read and update VTXIN");
  abi.stage.vertex_inputs = 0;
  Reject([&] { ValidateComputeProgram(program, abi); });
  Reject([&] { DecodePcoProgram(ShaderStage::kVertex, binary); });
  Reject([&] { DecodePcoProgram(ShaderStage::kFragment, binary); });
  binary[11] |= 1; // 原生 I_ONE_3B11I 的 index bit6 -> VTXIN64。
  Reject([&] { DecodeComputePcoProgram(binary); });
}
} // namespace

int main(int argc, char **argv) {
  try {
    TestNativePair(0); TestNativePair(1); TestNativeImmediateInput();
    // 可選：讀取外部完整原生 dump，只驗 decode，不捏造 shader 預期結果。
    for (int i = 1; i < argc; ++i) {
      std::ifstream input(argv[i], std::ios::binary);
      Check(input.good(), "cannot read native compute dump");
      const std::vector<std::uint8_t> binary{
          std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
      const auto decoded = DecodeComputePcoProgram(binary);
      Check(decoded.summary.ends_task, "native dump did not end its task");
      std::cout << argv[i] << ": " << decoded.instructions.size() << " groups PASS\n";
    }
    std::cout << "native compute ADD64_32 independent destination banks PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
