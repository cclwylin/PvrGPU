// SPDX-License-Identifier: MIT
#include "shader/compute_iss.h"
#include "shader/geometry_iss.h"
#include "shader/tessellation_iss.h"

#include "pco_find_top_bit_fixtures.h"
#include "pco_geometry_fixtures.h"
#include "pco_tessellation_patch_fixtures.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace pvrgpu::stub;

unsigned checks = 0;
void Check(bool value, const char *message) {
  ++checks;
  if (!value)
    throw std::runtime_error(message);
}
template <class Function>
void Reject(Function function, const char *message) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception &) {
    rejected = true;
  }
  Check(rejected, message);
}

const std::vector<std::uint8_t> kNopEnd{
    0x04, 0x80, 0xee, 0x00, 0xf2, 0xff, 0xff, 0xff};

// Unmodified compiler groups from the dEQP uaddCarry/usubBorrow captures.
const std::array<std::vector<std::uint8_t>, 4> kUaddGroups{{
    {0x99,0xd4,0x00,0xd4,0x3c,0xf2,0xa0,0x9c,0x1e,0x87,0xe0,
     0x84,0x40,0x30,0x88,0x01,0x81,0x41},
    {0x99,0xd4,0x00,0xd4,0x3c,0xf2,0xa0,0x9c,0x1e,0x87,0xe0,
     0xc2,0x40,0x30,0xc6,0x01,0x81,0x41},
    {0x99,0xd4,0x00,0xd4,0x3c,0xf2,0xa0,0x9c,0x1e,0x87,0xe0,
     0xc7,0x40,0x20,0x45,0x01,0x81,0x44},
    {0x99,0xd4,0x00,0xd4,0x3c,0xf2,0xa0,0x9c,0x1e,0x87,0xe0,
     0xc5,0x40,0x20,0x46,0x01,0x81,0x43},
}};
const std::array<std::vector<std::uint8_t>, 4> kUsubGroups{{
    {0x9a,0xd4,0x00,0xd4,0x3c,0xf2,0xa0,0x9c,0x1e,0x87,0xf0,0x10,
     0x88,0x40,0x30,0x84,0x01,0x81,0x45,0xff},
    {0x9a,0xd4,0x00,0xd4,0x3c,0xf2,0xa0,0x9c,0x1e,0x87,0xf0,0x10,
     0xc6,0x40,0x30,0xc2,0x01,0x81,0x45,0xff},
    {0x9a,0xd4,0x00,0xd4,0x3c,0xf2,0xa0,0x9c,0x1e,0x87,0xf0,0x10,
     0xc5,0x40,0x20,0x47,0x01,0x81,0x44,0xff},
    {0x9a,0xd4,0x00,0xd4,0x3c,0xf2,0xa0,0x9c,0x1e,0x87,0xf0,0x10,
     0xc6,0x40,0x20,0x45,0x01,0x81,0x43,0xff},
}};

std::vector<std::uint8_t> Prefix(std::vector<std::uint8_t> head,
                                 const std::vector<std::uint8_t> &tail) {
  head.insert(head.end(), tail.begin(), tail.end());
  return head;
}

PcoDecodedProgram DecodeCaptured(unsigned capture, bool subtract) {
  const auto &group = subtract ? kUsubGroups.at(capture)
                               : kUaddGroups.at(capture);
  switch (capture) {
  case 0:
    return DecodePcoProgram(ShaderStage::kVertex,
                            Prefix(group, FillSolidVertexPcoBinary()));
  case 1:
    return DecodePcoProgram(ShaderStage::kFragment,
                            Prefix(group, test::FindTopBitFixture(0)));
  case 2:
    return DecodeGeometryPcoProgram(Prefix(group, GeometryNativeLoadFixture()));
  case 3:
    return DecodeComputePcoProgram(Prefix(group, kNopEnd));
  default:
    throw std::runtime_error("bad capture index");
  }
}

void CheckDecoded(const PcoInstruction &instruction, PcoOpcode opcode,
                  PcoRegisterBank bank, unsigned source0, unsigned source1,
                  unsigned destination) {
  if (!(instruction.opcode == opcode && instruction.source_count == 2 &&
        instruction.repeat_count == 1 && !instruction.end_group &&
        instruction.source.bank == bank && instruction.source.index == source0 &&
        instruction.source1.bank == bank && instruction.source1.index == source1 &&
        instruction.target == PcoWriteTarget::kNone &&
        instruction.output_index == 0 &&
        instruction.output_target1 == PcoWriteTarget::kTemporary &&
        instruction.output_index1 == destination)) {
    std::cerr << "decoded expected/opcode/count/repeat/end/banks/sources/dest "
              << static_cast<unsigned>(opcode) << '/'
              << static_cast<unsigned>(instruction.opcode) << ' '
              << static_cast<unsigned>(instruction.source_count) << ' '
              << static_cast<unsigned>(instruction.repeat_count) << ' '
              << static_cast<unsigned>(instruction.end_group) << ' '
              << static_cast<unsigned>(instruction.source.bank) << ':'
              << instruction.source.index << ' '
              << static_cast<unsigned>(instruction.source1.bank) << ':'
              << instruction.source1.index << ' '
              << static_cast<unsigned>(instruction.target) << ':'
              << instruction.output_index << ' '
              << static_cast<unsigned>(instruction.output_target1) << ':'
              << instruction.output_index1 << '\n';
  }
  Check(instruction.opcode == opcode && instruction.source_count == 2 &&
            instruction.repeat_count == 1 && !instruction.end_group &&
            instruction.source.bank == bank &&
            instruction.source.index == source0 &&
            instruction.source1.bank == bank &&
            instruction.source1.index == source1 &&
            instruction.target == PcoWriteTarget::kNone &&
            instruction.output_index == 0 &&
            instruction.output_target1 == PcoWriteTarget::kTemporary &&
            instruction.output_index1 == destination,
        "captured carry/borrow semantic shape changed");
  const auto counts = CountPcoInstructions({instruction}, true);
  Check(counts.alu == 1 && counts.texture == 0 && counts.memory == 0,
        "carry/borrow is exactly one ALU issue");
}

void TestCapturedStages() {
  const std::array<PcoRegisterBank,4> banks{
      PcoRegisterBank::kVertexInput, PcoRegisterBank::kCoefficient,
      PcoRegisterBank::kTemporary, PcoRegisterBank::kTemporary};
  const std::array<unsigned,4> source0{4,2,7,5};
  const std::array<unsigned,4> source1{8,6,5,6};
  const std::array<unsigned,4> add_destination{1,1,4,3};
  const std::array<unsigned,4> sub_destination{5,5,4,3};
  for (unsigned capture = 0; capture < 4; ++capture) {
    const auto add = DecodeCaptured(capture, false);
    CheckDecoded(add.instructions.front(), PcoOpcode::kUnsignedAddCarry,
                 banks[capture], source0[capture], source1[capture],
                 add_destination[capture]);
    const auto sub = DecodeCaptured(capture, true);
    CheckDecoded(sub.instructions.front(), PcoOpcode::kUnsignedSubBorrow,
                 banks[capture], source0[capture], source1[capture],
                 sub_destination[capture]);
  }

  // The native task decoder is shared by compute/geometry/TCS/TES.  The real
  // geometry register form is valid unchanged in both tessellation stages.
  for (const auto stage : {ShaderStage::kTessellationControl,
                           ShaderStage::kTessellationEvaluation}) {
    const auto &tail = stage == ShaderStage::kTessellationControl
        ? kTessPatchFiveToTenTcs : kTessPatchFiveToTenTes;
    CheckDecoded(DecodeTessellationPcoProgram(
                     stage, Prefix(kUaddGroups[2], tail)).instructions.front(),
                 PcoOpcode::kUnsignedAddCarry,
                 PcoRegisterBank::kTemporary, 7, 5, 4);
    CheckDecoded(DecodeTessellationPcoProgram(
                     stage, Prefix(kUsubGroups[2], tail)).instructions.front(),
                 PcoOpcode::kUnsignedSubBorrow,
                 PcoRegisterBank::kTemporary, 7, 5, 4);
  }
}

std::vector<std::uint8_t> AuthoredDestinations(
    const std::vector<std::uint8_t> &captured, bool subtract,
    bool write0, bool write1, std::uint8_t destination0 = 0x80,
    std::uint8_t destination1 = 0x41) {
  Check(write0 || write1, "authored group needs a live destination");
  std::vector<std::uint8_t> bytes = captured;
  // Drop the captured single destination and its optional word padding,
  // retaining the complete fixed phase/source/ISS body.
  bytes.resize(subtract ? 18U : 17U);
  bytes[1] = static_cast<std::uint8_t>(0xd0U |
      (write0 ? 0x02U : 0U) | (write1 ? 0x04U : 0U));
  if (write0 && write1) {
    bytes.push_back(destination0);
    bytes.push_back(destination1);
  } else {
    // The header W bit identifies the physical slot; one live destination
    // always uses the ordinary single-destination grammar.
    bytes.push_back(0x43U);
  }
  if ((bytes.size() & 1U) != 0)
    bytes.push_back(0xffU);
  bytes[0] = static_cast<std::uint8_t>(0x90U | (bytes.size() / 2U));
  return bytes;
}

struct ArithmeticCase {
  std::uint32_t a;
  std::uint32_t b;
  std::uint32_t low;
  std::uint32_t flag;
};
const std::array<ArithmeticCase,3> kAddCases{{
    {UINT32_MAX, 1, 0, 1},
    {UINT32_MAX, UINT32_MAX, UINT32_C(0xfffffffe), 1},
    {UINT32_C(0x7fffffff), 1, UINT32_C(0x80000000), 0},
}};
const std::array<ArithmeticCase,3> kSubCases{{
    {0, 1, UINT32_MAX, 1},
    {1, 0, 1, 0},
    {UINT32_C(0x80000000), UINT32_MAX, UINT32_C(0x80000001), 1},
}};

ComputePcoAbi ComputeAbi() {
  ComputePcoAbi abi;
  abi.local_size = {1,1,1};
  abi.stage.temps = 7;
  return abi;
}

void TestPureAndComputeExecution() {
  for (const bool subtract : {false, true}) {
    const PcoOpcode opcode = subtract ? PcoOpcode::kUnsignedSubBorrow
                                      : PcoOpcode::kUnsignedAddCarry;
    const auto &cases = subtract ? kSubCases : kAddCases;
    PcoInstruction scalar;
    scalar.opcode = opcode;
    for (const auto &test : cases) {
      const auto value = EvaluatePcoCarryBorrow(opcode, test.a, test.b);
      Check(value.low == test.low && value.flag == test.flag,
            "carry/borrow dual-result boundary semantics");
      Check(EvaluatePcoAluInstruction(scalar, {test.a,test.b,0,0}) ==
                test.flag,
            "scalar carry/borrow evaluator returns the NIR W1 result");
    }

    for (unsigned live = 0; live < 3; ++live) {
      const bool write0 = live != 0;
      const bool write1 = live != 1;
      const auto group = AuthoredDestinations(
          subtract ? kUsubGroups[3] : kUaddGroups[3], subtract,
          write0, write1);
      if (live == 0)
        Check(group == (subtract ? kUsubGroups[3] : kUaddGroups[3]),
              "authored W1-only form differs from real compiler bytes");
      const auto program = DecodeComputePcoProgram(Prefix(group, kNopEnd));
      const auto &instruction = program.instructions.front();
      Check((instruction.target != PcoWriteTarget::kNone) == write0 &&
                (instruction.output_target1 != PcoWriteTarget::kNone) == write1,
            "header W0/W1 liveness did not reach semantic destinations");
      const auto abi = ComputeAbi();
      ValidateComputeProgram(program, abi);
      for (const auto &test : cases) {
        auto task = MakeComputeTask(abi, {}, {1,1,1}, {0,0,0}, 0, 1);
        for (auto &lane : task.lanes) {
          lane.temporaries[5] = test.a;
          lane.temporaries[6] = test.b;
          lane.temporary_written.set(5);
          lane.temporary_written.set(6);
        }
        ComputeWorkgroupResult stats;
        StepComputeTask(program, abi, task, {}, stats);
        if (write0)
          Check(task.lanes[0].temporaries[live == 1 ? 3 : 0] == test.low,
                "compute W0 low word is incorrect");
        if (write1)
          Check(task.lanes[0].temporaries[live == 0 ? 3 : 1] == test.flag,
                "compute W1 flag is incorrect");
        Check(stats.stats.alu_instructions == kComputeTaskWidth,
              "compute carry/borrow instruction counter changed");
      }
    }

    // Both physical outputs overwrite the two input registers in the opposite
    // order.  Passing this requires all inputs/results to be sampled before W0.
    const auto alias_group = AuthoredDestinations(
        subtract ? kUsubGroups[3] : kUaddGroups[3], subtract, true, true,
        0x86U, 0x45U); // W0=t6, W1=t5.
    const auto alias = DecodeComputePcoProgram(Prefix(alias_group, kNopEnd));
    ValidateComputeProgram(alias, ComputeAbi());
    for (const auto &test : cases) {
      auto task = MakeComputeTask(ComputeAbi(), {}, {1,1,1}, {0,0,0}, 0, 1);
      for (auto &lane : task.lanes) {
        lane.temporaries[5] = test.a;
        lane.temporaries[6] = test.b;
        lane.temporary_written.set(5);
        lane.temporary_written.set(6);
      }
      ComputeWorkgroupResult stats;
      StepComputeTask(alias, ComputeAbi(), task, {}, stats);
      Check(task.lanes[0].temporaries[6] == test.low &&
                task.lanes[0].temporaries[5] == test.flag,
            "dual carry/borrow input/output alias corrupted a result");
    }
  }
}

void TestNativeStageDualWrites() {
  for (const bool subtract : {false, true}) {
    const auto &test = (subtract ? kSubCases : kAddCases).front();
    const auto both = AuthoredDestinations(
        subtract ? kUsubGroups[2] : kUaddGroups[2], subtract, true, true);

    auto geometry = DecodeGeometryPcoProgram(
        Prefix(both, GeometryNativeLoadFixture()));
    auto geometry_abi = GeometryNativeLoadAbi();
    geometry_abi.temps = 8;
    ValidateGeometryProgram(geometry, geometry_abi);
    auto geometry_task = MakeGeometryTask(
        geometry_abi, std::vector<std::uint32_t>(4), 0, 0);
    geometry_task.temporaries[7] = test.a;
    geometry_task.temporaries[5] = test.b;
    geometry_task.temporary_written.set(7);
    geometry_task.temporary_written.set(5);
    GeometryExecutionStats geometry_stats;
    StepGeometryTask(geometry, geometry_abi, geometry_task, {}, geometry_stats);
    Check(geometry_task.temporaries[0] == test.low &&
              geometry_task.temporaries[1] == test.flag,
          "geometry did not explicitly commit both carry results");

    for (const auto stage : {ShaderStage::kTessellationControl,
                             ShaderStage::kTessellationEvaluation}) {
      const bool control = stage == ShaderStage::kTessellationControl;
      const auto &tail = control ? kTessPatchFiveToTenTcs
                                 : kTessPatchFiveToTenTes;
      const auto program = DecodeTessellationPcoProgram(
          stage, Prefix(both, tail));
      DriverPcoStageAbi abi;
      abi.temps = 8;
      abi.vertex_inputs = control ? 3 : 5;
      abi.vertex_outputs = control ? 0 : 4;
      abi.shareds = abi.uniform_buffer_descriptor_start =
          abi.push_constant_start = control ? 8 : 4;
      ValidateTessellationProgram(program, abi);
      const std::array<std::uint32_t,3> coordinates{};
      auto task = control
          ? MakeTessellationControlTask(
                abi, std::vector<std::uint32_t>(8), 0, 5, 10)
          : MakeTessellationEvaluationTask(
                abi, std::vector<std::uint32_t>(4), 0, 10,
                &coordinates, 1);
      for (unsigned lane = 0; lane < task.lane_count; ++lane) {
        task.lanes[lane].temporaries[7] = test.a;
        task.lanes[lane].temporaries[5] = test.b;
        task.lanes[lane].temporary_written.set(7);
        task.lanes[lane].temporary_written.set(5);
      }
      TessellationExecutionStats stats;
      StepTessellationTask(program, abi, task, {}, stats);
      for (unsigned lane = 0; lane < task.lane_count; ++lane)
        Check(task.lanes[lane].temporaries[0] == test.low &&
                  task.lanes[lane].temporaries[1] == test.flag,
              "tessellation did not explicitly commit both carry results");
    }
  }
}

void TestStrictDecodeAndValidation() {
  for (const bool subtract : {false, true}) {
    const auto &original = subtract ? kUsubGroups[3] : kUaddGroups[3];
    const auto reject_group = [&](const std::vector<std::uint8_t> &group) {
      Reject([&] { DecodeComputePcoProgram(Prefix(group, kNopEnd)); },
             "noncanonical carry/borrow group was accepted");
    };
    for (unsigned offset = 3; offset <= (subtract ? 11U : 10U); ++offset) {
      auto corrupt = original;
      corrupt[offset] ^= 1U;
      reject_group(corrupt);
    }
    for (unsigned mutation = 0; mutation < 9; ++mutation) {
      auto corrupt = original;
      const unsigned lower = subtract ? 12U : 11U;
      switch (mutation) {
      case 0: corrupt[0] ^= 0x10U; break;       // DA
      case 1: corrupt[1] ^= 0x08U; break;       // OLCHK
      case 2: corrupt[1] &= ~0x04U; break;      // no live output
      case 3: corrupt[1] |= 0x01U; break;       // CC
      case 4: corrupt[2] |= 0x80U; break;       // END
      case 5: corrupt[2] |= 0x02U; break;       // repeat 2
      case 6: corrupt[lower + 1] ^= 0x01U; break; // s1 != sc0
      case 7: corrupt[lower + 2] ^= 0x20U; break; // is0 != s3
      case 8: corrupt[lower + 4] ^= 0x01U; break; // upper != sc1
      }
      reject_group(corrupt);
    }
    {
      auto corrupt = original;
      corrupt[subtract ? 17U : 16U] ^= 1U;
      reject_group(corrupt); // ISS
    }
    {
      auto corrupt = original;
      corrupt[subtract ? 18U : 17U] = 0x01U;
      reject_group(corrupt); // non-writable special destination
    }
    if (subtract) {
      auto corrupt = original;
      corrupt.back() = 0;
      reject_group(corrupt);
    }
    for (std::size_t size = 3; size < original.size(); ++size) {
      auto truncated = original;
      truncated.resize(size);
      Reject([&] { DecodeComputePcoProgram(truncated); },
             "truncated carry/borrow group was accepted");
    }

    auto valid = DecodeComputePcoProgram(Prefix(original, kNopEnd));
    ValidateComputeProgram(valid, ComputeAbi());
    for (unsigned mutation = 0; mutation < 7; ++mutation) {
      auto bad = valid;
      auto &instruction = bad.instructions.front();
      switch (mutation) {
      case 0: instruction.source_count = 1; break;
      case 1: instruction.repeat_count = 2; break;
      case 2: instruction.output_target1 = PcoWriteTarget::kNone;
              instruction.output_index1 = 0; break;
      case 3: instruction.output_target1 = PcoWriteTarget::kPixelOutput; break;
      case 4: instruction.output_index1 = 7; break;
      case 5: instruction.target = PcoWriteTarget::kNone;
              instruction.output_index = 1; break;
      case 6: instruction.end_group = 1; break;
      }
      Reject([&] { ValidateComputeProgram(bad, ComputeAbi()); },
             "malformed carry/borrow semantic metadata was accepted");
    }
  }
}
} // namespace

int main() {
  try {
    TestCapturedStages();
    TestPureAndComputeExecution();
    TestNativeStageDualWrites();
    TestStrictDecodeAndValidation();
    std::cout << "PCO UADDC/USUBB: " << checks << " checks PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "PCO UADDC/USUBB after " << checks
              << " checks: " << error.what() << '\n';
    return 1;
  }
}
