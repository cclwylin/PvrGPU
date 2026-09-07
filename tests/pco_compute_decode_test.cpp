/* SPDX-License-Identifier: MIT */
#include "shader/pco_iss.h"
#include "pco_compute_fixtures.h"
#include "compute_atomic_reference.h"
#include <iostream>
#include <stdexcept>
using namespace pvrgpu::stub;
static void Check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
template <class Fn> static void Reject(Fn fn) {
  try { fn(); } catch (const std::exception &) { return; }
  throw std::runtime_error("expected fail-closed compute decode");
}
int main() {
  try {
    for (unsigned kind = 0; kind < 21; ++kind) {
      const auto d = DecodeComputePcoProgram(ComputePcoFixture(kind));
      Check(d.summary.stage == ShaderStage::kCompute && d.summary.ends_task &&
            !d.summary.pixel_output_mask && !d.summary.vertex_output_mask,
            "compute summary is independent of graphics outputs");
      unsigned stores = 0, loads = 0, masks = 0, predicates = 0, atomics = 0;
      for (const auto &i : d.instructions) {
        stores += i.opcode == PcoOpcode::kBufferStore;
        loads += i.opcode == PcoOpcode::kBufferLoad;
        masks += i.opcode == PcoOpcode::kConditionalMask;
        predicates += i.writes_predicate;
        atomics += IsPcoAtomic32(i.opcode);
        if (IsPcoAtomic32(i.opcode)) {
          Check(i.source_count == 3 && i.source.index + 1 == i.source1.index &&
                i.source.index + 2 == i.source2.index && i.component_count == 1 &&
                i.data_request == 0 && i.target == PcoWriteTarget::kTemporary,
                "native atomic ADD32 retains address/data triplet and old-value destination");
          Check(d.instructions.at(i.group_index + 1).opcode == PcoOpcode::kWaitDataFence,
                "atomic old-value result is protected by native WDF");
        }
        if (i.opcode == PcoOpcode::kBufferStore) {
          Check(i.source.index + 1 == i.source1.index && i.source_count == 3 &&
                i.component_count == (kind == 3 ? 3 : 1), "ST32 true native operands");
        }
        if (i.opcode == PcoOpcode::kBufferLoad)
          Check(i.memory_cache_mode == (kind == 5 ? 1 : 0), "native coherent LD cache modifier");
        if (i.opcode == PcoOpcode::kBranch && (kind < 8 || kind >= 10))
          Check(d.instructions[i.branch_target_index].opcode == PcoOpcode::kConditionalMask,
                "branch points to true guard epilogue");
      }
      Check(stores == (kind && kind != 12 ? 1 : 0) &&
            loads == ((kind == 1 || kind == 2 || kind == 5 || (kind >= 7 && kind < 10) || kind == 12 || kind >= 13) ? 1 : 0) &&
            masks == (kind == 12 ? 0 : (kind == 8 || kind == 9) ? 8 : kind ? 3 : 0) &&
            predicates == ((kind == 8 || kind == 9) ? 3 : kind ? 1 : 0) &&
            atomics == (kind >= 10 ? 1 : 0),
            "native compute instruction classes");
      CountPcoInstructions(d.instructions, true);
      Reject([&] { DecodePcoProgram(ShaderStage::kVertex, ComputePcoFixture(kind)); });
      Reject([&] { DecodePcoProgram(ShaderStage::kFragment, ComputePcoFixture(kind)); });
      std::cout << "compute fixture " << kind << ": " << d.instructions.size() << " groups PASS\n";
    }
    const auto cas = DecodeComputePcoProgram(ComputePcoFixture(21));
    unsigned mutexes = 0, instances = 0;
    for (const auto &instruction : cas.instructions) {
      Check(!IsPcoAtomic32(instruction.opcode), "CAS invented a DMA AMO instruction");
      if (instruction.opcode == PcoOpcode::kMutex) {
        Check(instruction.immediate == 0 && instruction.control_operation == (mutexes ? 0U : 3U),
              "native CAS lock/release order or mutex ID changed");
        ++mutexes;
        for (const unsigned invalid : {0x10U,0x20U,0x40U,0x80U}) {
          auto corrupt = ComputePcoFixture(21);
          // MUTEX payload follows its 3-byte header. Reserved bits and
          // release-sleep/wakeup are not quietly treated as a plain release.
          corrupt[instruction.binary_offset + 3] = invalid;
          Reject([&] { DecodeComputePcoProgram(corrupt); });
        }
      }
      instances += instruction.source.bank == PcoRegisterBank::kSpecial &&
                   instruction.source.index == kPcoSpecialInstanceNumber;
    }
    Check(mutexes == 2 && instances == 32, "CAS does not retain real mutex/per-instance lowering");
    Check(!PcoSpecialConstantBits(kPcoSpecialInstanceNumber,nullptr),
          "runtime INST_NUM was incorrectly added to the static constant table");
    Reject([&] { DecodePcoProgram(ShaderStage::kVertex, ComputePcoFixture(21)); });
    Reject([&] { DecodePcoProgram(ShaderStage::kFragment, ComputePcoFixture(21)); });
    const auto d = DecodeComputePcoProgram(ComputePcoFixture(1));
    const auto &comparison = d.instructions[0];
    Check(EvaluatePcoAluInstruction(comparison, {29, 30, 0, 0}) == UINT32_MAX &&
          EvaluatePcoAluInstruction(comparison, {30, 30, 0, 0}) == 0, "instance guard signed comparison");
    const auto &predicate = d.instructions[2];
    Check(EvaluatePcoPredicate(predicate, {0, 0, 0, 0}) &&
          !EvaluatePcoPredicate(predicate, {UINT32_MAX, 0, 0, 0}), "predicate is genuine TST result");
    const auto ids = DecodeComputePcoProgram(ComputePcoFixture(3));
    bool high = false;
    for (const auto &i : ids.instructions) {
      if (i.opcode != PcoOpcode::kIntegerMultiplyAdd64High) continue;
      high = true;
      Check(EvaluatePcoAluInstruction(i, {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX}) ==
             UINT32_C(0xfffffffe), "IMADD64 true modulo-64 high result");
    }
    Check(high, "non-power-of-two ID lowering uses native multiply-high");
    for (unsigned kind : {8U, 9U}) {
    const auto loop = DecodeComputePcoProgram(ComputePcoFixture(kind));
    unsigned setmask = 0, looptest = 0, backedge = 0;
    for (const auto &i : loop.instructions) {
      if (i.opcode == PcoOpcode::kConditionalMask && i.control_operation == 2) {
        ++setmask;
        Check(i.source_count == 2 && !i.immediate &&
              i.source1.bank == PcoRegisterBank::kSpecial && i.source1.index == 2,
              "CNDSM preserves real s2 set-mask operand, not immediate adjust");
      }
      if (i.opcode == PcoOpcode::kConditionalMask && i.control_operation == 3) {
        ++looptest;
        Check(i.source_count == 1 && i.immediate == 2 && i.writes_predicate,
              "CNDLT has native adjust and P0 output");
      }
      if (i.opcode == PcoOpcode::kBranch && i.branch_target_index < i.group_index) {
        ++backedge;
        Check(i.exec_cnd == 1 && i.branch_condition == 0,
              "native loop backedge is BR.CC under P0");
      }
    }
    Check(setmask == 1 && looptest == 1 && backedge == 1,
          "compiler retains actual dynamic loop and break groups");
    }
    auto bytes = ComputePcoFixture(1);
    bytes[0x36] = 1; // BR offset is no longer word aligned.
    Reject([&] { DecodeComputePcoProgram(bytes); });
    bytes = ComputePcoFixture(1); bytes[0xa5] |= 0x80; // ST.tiled not yet supported.
    Reject([&] { DecodeComputePcoProgram(bytes); });
    bytes = ComputePcoFixture(0); bytes[2] &= 0x7f; // no END.
    Reject([&] { DecodeComputePcoProgram(bytes); });
    const auto atomic = DecodeComputePcoProgram(ComputePcoFixture(10));
    for (const auto &i : atomic.instructions) {
      if (i.opcode != PcoOpcode::kAtomicAdd32) continue;
      // Exercise the documented ISA bitfield, not fabricated compiler output.
      // SUB exists in I_ATOMIC although NIR has no direct atomic-isub op.
      const std::array<PcoOpcode,10> expected{
          PcoOpcode::kAtomicAdd32, PcoOpcode::kAtomicSub32,
          PcoOpcode::kAtomicExchange32, PcoOpcode::kAtomicUnsignedMin32,
          PcoOpcode::kAtomicSignedMin32, PcoOpcode::kAtomicUnsignedMax32,
          PcoOpcode::kAtomicSignedMax32, PcoOpcode::kAtomicAnd32,
          PcoOpcode::kAtomicOr32, PcoOpcode::kAtomicXor32};
      for (unsigned index = 0; index < expected.size(); ++index) {
        bytes = ComputePcoFixture(10);
        bytes[i.binary_offset + 1] = kComputeAtomicNibbles[index] << 4;
        const auto decoded = DecodeComputePcoProgram(bytes);
        Check(decoded.instructions[i.group_index].opcode == expected[index],
              "native atomic nibble decoded to the wrong operation");
        Check(CountPcoInstructions(decoded.instructions, true).memory != 0,
              "native atomic instruction lost its memory classification");
      }
      for (unsigned reserved : {3U, 11U, 12U, 13U, 14U, 15U}) {
        bytes = ComputePcoFixture(10);
        bytes[i.binary_offset + 1] = reserved << 4;
        Reject([&] { DecodeComputePcoProgram(bytes); });
      }
      for (unsigned byte : {0U, 1U, 2U}) {
        bytes = ComputePcoFixture(10);
        bytes[i.binary_offset + byte] ^= 8U; // DRC1 / byte1 reserved / byte2 reserved.
        Reject([&] { DecodeComputePcoProgram(bytes); });
      }
    }
    std::cout << "compute decode/pure ALU regression PASS\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
