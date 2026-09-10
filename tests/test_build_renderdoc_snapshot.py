#!/usr/bin/env python3
"""Private builder protocol tests; mock compilers are NOT renderer evidence."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "script"
sys.path.insert(0, str(SCRIPT))
import build_renderdoc_snapshot as builder


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DebugMessagePartition(unittest.TestCase):
    """Execute the actual patch block with instrumented C++ message containers.

    This is a host-algorithm test, not a GL replay or renderer test. Set
    PVRGPU_SNAPSHOT_TEST_SANITIZERS=1 for an additional ASan/UBSan execution.
    Select a configured toolchain with PVRGPU_SNAPSHOT_TEST_CXX or CXX;
    macOS hosts may need the Homebrew LLVM path instead of Apple clang++.
    """
    def test_actual_partition_preserves_fields_order_and_linear_work(self):
        compiler = (os.environ.get("PVRGPU_SNAPSHOT_TEST_CXX") or os.environ.get("CXX") or
                    shutil.which("clang++"))
        if not compiler:
            self.skipTest("clang++ is required for the actual C++ partition regression")
        patch = (SCRIPT.parent / "third_party/renderdoc-drawlist-snapshot.patch").read_text()
        section = next(s for s in patch.split("diff --git ")[1:]
                       if s.startswith("a/renderdoc/driver/gl/gl_driver.cpp "))
        updated = "\n".join(line[1:] for line in section.splitlines()
                            if line.startswith(("+", " ")) and not line.startswith("+++"))
        self.assertIn("if(ser.IsReading() && IsLoading(m_State) && m_ReplayOptions.apiValidation)", updated)
        start = updated.index("    DebugMessages.clear();")
        end = updated.index("    m_DebugMessages.swap(attributedMessages);", start)
        block = updated[start:end + len("    m_DebugMessages.swap(attributedMessages);")]
        self.assertNotIn("removeIf", "\n".join(x for x in block.splitlines() if not x.lstrip().startswith("//")))
        program = r'''
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
static uint64_t copies=0, moves=0, checks=0;
struct DebugMessage {
  uint32_t eventId, category, severity, source, messageID;
  std::string description;
  DebugMessage(uint32_t i, uint32_t event): eventId(event),category(i%8),severity(i%5),
    source(i%7),messageID(i),description(std::string("message\0",8)+std::to_string(i)) {}
  DebugMessage(const DebugMessage &o): eventId(o.eventId),category(o.category),severity(o.severity),
    source(o.source),messageID(o.messageID),description(o.description) { ++copies; }
#ifndef COPY_ONLY_MESSAGE
  DebugMessage(DebugMessage &&o) noexcept: eventId(o.eventId),category(o.category),severity(o.severity),
    source(o.source),messageID(o.messageID),description(std::move(o.description)) { ++moves; }
#endif
  bool operator==(const DebugMessage &o) const {
    return eventId==o.eventId && category==o.category && severity==o.severity &&
      source==o.source && messageID==o.messageID && description==o.description;
  }
};
template<class T> using rdcarray=std::vector<T>;
static void check(bool value) { ++checks; if(!value) throw std::runtime_error("partition mismatch"); }
static void partition(rdcarray<DebugMessage> &m_DebugMessages,
                      rdcarray<DebugMessage> &DebugMessages) {
ACTUAL_BLOCK
}
int main() {
  try {
    for(unsigned count : {0U,1U,2U,3U,17U,1024U,100000U})
    for(unsigned mode=0;mode<4;++mode)
    for(unsigned captured : {0U,3U,19U}) {
      rdcarray<DebugMessage> history, serialized, expectedHistory, expectedNew;
      for(unsigned i=0;i<count;++i) {
        const uint32_t event=mode==0 ? 0 : mode==1 ? i+1 :
          mode==2 ? (i%2 ? i+1 : 0) : (i%5 ? i+1 : 0);
        history.emplace_back(i,event);
        if(event) expectedHistory.push_back(history.back()); else expectedNew.push_back(history.back());
      }
      for(unsigned i=0;i<captured;++i) serialized.emplace_back(0xf0000000U+i,0);
      copies=moves=0;
      partition(history,serialized);
      // Reserve before moving means one construction per original message.
      // Upstream DebugMessage is copy-only; test both that and movable types.
      check(copies+moves==count);
      check(history==expectedHistory);
      check(serialized==expectedNew);
      check(history.size()+serialized.size()==count);
    }
    std::printf("PASS %llu checks; stable complete messages, captured replacement, linear constructions\n",
                static_cast<unsigned long long>(checks));
  } catch(const std::exception &e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
'''.replace("ACTUAL_BLOCK", block)
        with tempfile.TemporaryDirectory(prefix="snapshot-debug-partition-") as directory:
            path = Path(directory)
            cpp = path / "actual-partition.cpp"
            cpp.write_text(program)
            modes = [("normal", ["-O2"])]
            if os.environ.get("PVRGPU_SNAPSHOT_TEST_SANITIZERS") == "1":
                modes.append(("sanitized", ["-O1", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]))
            for mode, flags in modes:
                for copy_only in (False, True):
                    with self.subTest(mode=mode, copy_only=copy_only):
                        exe = path / (mode + ("-copy" if copy_only else "-move"))
                        command = [*shlex.split(compiler), "-std=c++14", "-Wall", "-Wextra", "-Werror", *flags]
                        if copy_only:
                            command.append("-DCOPY_ONLY_MESSAGE")
                        built = subprocess.run([*command, str(cpp), "-o", str(exe)], text=True,
                                               capture_output=True)
                        self.assertEqual(built.returncode, 0,
                            "Actual partition compilation failed; set PVRGPU_SNAPSHOT_TEST_CXX or CXX "
                            "to the configured C++ toolchain (e.g. Homebrew LLVM on macOS).\n" +
                            built.stdout + built.stderr)
                        result = subprocess.run([str(exe)], text=True, capture_output=True)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn("PASS 336 checks", result.stdout)


class ReplayCapabilities(unittest.TestCase):
    """Execute actual patched guards with host GL mocks, not a renderer/ABI test."""
    @staticmethod
    def updated(name):
        patch = (SCRIPT.parent / "third_party/renderdoc-drawlist-snapshot.patch").read_text()
        section = next(s for s in patch.split("diff --git ")[1:] if s.startswith("a/" + name + " "))
        return "\n".join(line[1:] for line in section.splitlines()
                         if line.startswith(("+", " ")) and not line.startswith("+++"))

    @staticmethod
    def function(source, marker):
        start = source.index(marker)
        at = source.index("{", start)
        depth = 1
        end = at + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return source[start:end]

    def test_actual_source_preflight_and_error_propagation_order(self):
        common = self.updated("renderdoc/driver/gl/gl_common.cpp")
        fetch = common[common.index("bool FetchEnabledExtensions()") :]
        self.assertLess(fetch.index("const GLenum pendingError = GL.glGetError();"),
                        fetch.index("GetContextVersion(ctxGLES, ctxVersion);"))
        self.assertIn("return ValidateGLESReplayCapabilities();", common)
        capability = self.function(common, "static bool ValidateGLESReplayCapabilities()")
        self.assertNotIn("ClearGLErrors", capability)
        self.assertIn("error == eGL_INVALID_ENUM", capability)
        self.assertNotIn("eGL_INVALID_OPERATION", capability)
        state = self.updated("renderdoc/driver/gl/gl_renderstate.cpp")
        apply = state[state.index("void GLRenderState::ApplyState(WrappedOpenGL *driver)"):]
        self.assertLess(apply.index("driver->m_FatalError != ResultCode::Succeeded"),
                        apply.index("ReplayCapabilityError()"))
        self.assertLess(apply.index("ReplayCapabilityError()"),
                        apply.index("for(GLuint i = 0; i < eEnabled_Count; i++)"))
        replay = self.updated("renderdoc/driver/gl/gl_replay.cpp")
        failed = replay[replay.index("if(!FetchEnabledExtensions())"):]
        self.assertLess(failed.index("platform.DeleteReplayContext(data);"),
                        failed.index('"Replay capability discovery failed"'))
        driver = self.updated("renderdoc/driver/gl/gl_driver.cpp")
        # ApplyState itself is unchanged and outside this patch hunk. Check
        # the post-apply message restoration/fatal propagation that it contains.
        begin = driver[driver.index("// restore saved messages - which implicitly discards any generated while applying state"):]
        self.assertLess(begin.index("savedDebugMessages.swap(m_DebugMessages);"),
                        begin.index("if(m_FatalError != ResultCode::Succeeded)"))
        self.assertIn("m_FailedReplayResult = m_FatalError;\n      return false;", begin)
        snapshot = self.updated("renderdoc/driver/gl/gl_replay_snapshot.inl")
        decode = self.function(snapshot, "static void DecodeContextState(")
        self.assertLess(decode.index("state.ReplayCapabilityError()"),
                        decode.index("ContextState(d, state, extra) == blob"))
        load = snapshot[snapshot.index("DecodeContextState(d, stateBlob, state, extra);"):]
        self.assertLess(load.index("DecodeContextState(d, stateBlob, state, extra);"),
                        load.index("RestoreResource(d, record)"))
        self.assertIn('Require(d.FatalErrorCheck() == ResultCode::Succeeded, "snapshot restored state capability check failed")', load)

    def test_actual_capability_and_state_guards(self):
        compiler = (os.environ.get("PVRGPU_SNAPSHOT_TEST_CXX") or os.environ.get("CXX") or
                    shutil.which("clang++"))
        if not compiler:
            self.skipTest("C++ compiler required for extracted capability guards")
        common = self.updated("renderdoc/driver/gl/gl_common.cpp")
        state = self.updated("renderdoc/driver/gl/gl_renderstate.cpp")
        capability = self.function(common, "static bool ValidateGLESReplayCapabilities()")
        checker = self.function(state, "const char *GLRenderState::ReplayCapabilityError() const")
        start = state.index("void GLRenderState::ApplyState(WrappedOpenGL *driver)")
        apply = state[start:state.index("  for(GLuint i = 0; i < eEnabled_Count; i++)", start)]
        # Execute the complete real preflight prefix. The first side effect of
        # the remaining ApplyState body is represented by one mutation count.
        apply += "  ++driver->mutations;\n}"
        program = r'''
#include <array>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <stdexcept>
#include <string>
using GLenum=unsigned; using GLint=int; using GLuint=unsigned; using GLboolean=uint8_t;
enum : unsigned {eGL_NONE=0,eGL_INVALID_ENUM=0x500,eGL_INVALID_VALUE=0x501,
  eGL_INVALID_OPERATION=0x502,eGL_OUT_OF_MEMORY=0x505,eGL_CONTEXT_LOST=0x507,
  eGL_TEXTURE_BINDING_BUFFER=0x8c2c,eGL_BLEND_EQUATION_RGB=0x8009,
  eGL_COLOR_WRITEMASK=0xc23,eGL_MAX_DRAW_BUFFERS=0x8824};
enum {ARB_texture_buffer_object,ARB_texture_buffer_range,ARB_draw_buffers_blend,EXT_draw_buffers2};
static std::array<bool,4> HasExt;
static bool IsGLES=true; static int GLCoreVersion=32;
static uint64_t checks=0; static unsigned queries=0,errorReads=0;
static std::array<GLenum,3> errors; static std::deque<GLenum> pending;
static int extraAt=-1,maxDraws=8; static GLenum limitError=eGL_NONE;
template<class... A> static void logMessage(const char *, A...) {}
#define RDCWARN(...) logMessage(__VA_ARGS__)
#define RDCERR(...) logMessage(__VA_ARGS__)
#define ARRAY_COUNT(a) (sizeof(a)/sizeof((a)[0]))
static void check(bool value) { ++checks; if(!value) throw std::runtime_error("capability guard mismatch"); }
static GLenum getError() { ++errorReads; if(pending.empty())return eGL_NONE;
  GLenum e=pending.front();pending.pop_front();return e; }
static void finishQuery(unsigned i) { ++queries; if(errors[i])pending.push_back(errors[i]);
  if(extraAt==int(i))pending.push_back(eGL_OUT_OF_MEMORY); }
static void getInteger(GLenum pname,GLint *value) {
  if(pname==eGL_MAX_DRAW_BUFFERS){++queries;*value=maxDraws;if(limitError)pending.push_back(limitError);}
  else {check(pname==eGL_TEXTURE_BINDING_BUFFER);*value=0;finishQuery(0);} }
static void getIndexedInteger(GLenum pname,GLuint index,GLint *value) {
  check(pname==eGL_BLEND_EQUATION_RGB && index==0);*value=0x8006;finishQuery(1); }
static void getIndexedBoolean(GLenum pname,GLuint index,GLboolean *value) {
  check(pname==eGL_COLOR_WRITEMASK && index==0);for(unsigned i=0;i<4;++i)value[i]=1;finishQuery(2); }
struct Functions { GLenum (*glGetError)(); void (*glGetIntegerv)(GLenum,GLint*);
  void (*glGetIntegeri_v)(GLenum,GLuint,GLint*);void (*glGetBooleani_v)(GLenum,GLuint,GLboolean*); } GL;
static void reset() { HasExt={{true,true,true,true}};IsGLES=true;GLCoreVersion=32;
  GL={getError,getInteger,getIndexedInteger,getIndexedBoolean};errors={{0,0,0}};
  pending.clear();extraAt=-1;maxDraws=8;limitError=0;queries=errorReads=0; }
ACTUAL_CAPABILITY
enum class ResultCode {Succeeded,APIHardwareUnsupported,ExistingFailure};
struct ContextPair {void *ctx=reinterpret_cast<void*>(uintptr_t(1));};
struct WrappedOpenGL {ResultCode m_FatalError=ResultCode::Succeeded; ContextPair context;
  unsigned mutations=0; ContextPair &GetCtx(){return context;} };
static void setError(ResultCode &target,ResultCode value,const char *,const char *text) {
  check(text!=nullptr);target=value; }
#define SET_ERROR_RESULT(...) setError(__VA_ARGS__)
struct GLRenderState {
  bool ContextPresent=true;
  struct Binding {unsigned name=0;}; Binding TexBuffer[128],BufferBindings[16];
  enum {eBufIdx_Texture=5};
  struct Blend {GLenum EquationRGB=0,EquationAlpha=0,SourceRGB=0,SourceAlpha=0,
    DestinationRGB=0,DestinationAlpha=0;bool Enabled=false;} Blends[8];
  struct Mask {GLboolean red=0,green=0,blue=0,alpha=0;} ColorMasks[8];
  const char *ReplayCapabilityError() const;void ApplyState(WrappedOpenGL *);
};
ACTUAL_CHECKER
ACTUAL_APPLY
static GLRenderState uniform(unsigned count=8) { GLRenderState s;
  for(unsigned i=0;i<count;++i){s.Blends[i]={0x8006,0x8006,1,1,0,0,false};s.ColorMasks[i]={1,1,1,1};}
  return s; }
static void blendChange(GLRenderState::Blend &b,unsigned f) {
  switch(f){case 0:++b.EquationRGB;break;case 1:++b.EquationAlpha;break;
    case 2:++b.SourceRGB;break;case 3:++b.SourceAlpha;break;
    case 4:++b.DestinationRGB;break;case 5:++b.DestinationAlpha;break;default:b.Enabled=!b.Enabled;}
}
static void maskChange(GLRenderState::Mask &m,unsigned f) {
  switch(f){case 0:m.red=!m.red;break;case 1:m.green=!m.green;break;
    case 2:m.blue=!m.blue;break;default:m.alpha=!m.alpha;}
}
static void reject(const GLRenderState &s,const char *wanted) {
  const char *actual=s.ReplayCapabilityError();check(actual && std::string(actual)==wanted); }
int main() {
 try {
  // All three independent success/INVALID_ENUM combinations, including the
  // blend-success/mask-failure OR-gate hazard; updates commit transactionally.
  for(unsigned bits=0;bits<8;++bits){reset();for(unsigned i=0;i<3;++i)errors[i]=(bits&(1U<<i))?eGL_INVALID_ENUM:eGL_NONE;
    check(ValidateGLESReplayCapabilities());check(queries==3 && pending.empty());
    check(HasExt[0]==!(bits&1));check(HasExt[1]==!(bits&1));
    check(HasExt[2]==(!(bits&2)&&!(bits&4)));check(HasExt[3]==!(bits&4));}
  for(unsigned bits=0;bits<16;++bits){reset();for(unsigned i=0;i<4;++i)HasExt[i]=(bits>>i)&1;
    const auto before=HasExt;check(ValidateGLESReplayCapabilities());
    check(queries==unsigned(before[0])+unsigned(before[2])+unsigned(before[2]||before[3]));
    check(HasExt[0]==before[0] && HasExt[1]==(before[0]&&before[1]) && HasExt[2]==before[2] && HasExt[3]==before[3]);}
  for(unsigned at=0;at<3;++at)for(GLenum bad:{eGL_INVALID_OPERATION,eGL_INVALID_VALUE,eGL_OUT_OF_MEMORY,eGL_CONTEXT_LOST}){
    reset();for(unsigned i=0;i<at;++i)errors[i]=eGL_INVALID_ENUM;errors[at]=bad;
    const auto before=HasExt;check(!ValidateGLESReplayCapabilities());check(HasExt==before);check(queries==at+1);}
  for(int at=0;at<3;++at){reset();extraAt=at;const auto before=HasExt;
    check(!ValidateGLESReplayCapabilities());check(HasExt==before);check(queries==unsigned(at+1));}
  for(unsigned missing=0;missing<4;++missing){reset();const auto before=HasExt;
    errors={{eGL_INVALID_ENUM,eGL_INVALID_ENUM,eGL_INVALID_ENUM}};
    if(missing==0)GL.glGetError=nullptr;if(missing==1)GL.glGetIntegerv=nullptr;
    if(missing==2)GL.glGetIntegeri_v=nullptr;if(missing==3)GL.glGetBooleani_v=nullptr;
    check(!ValidateGLESReplayCapabilities());check(HasExt==before);}
  reset();pending.push_back(eGL_INVALID_ENUM);auto before=HasExt;
  check(!ValidateGLESReplayCapabilities());check(HasExt==before && queries==0);
  for(unsigned bypass=0;bypass<2;++bypass){reset();GL.glGetError=nullptr;
    if(bypass)GLCoreVersion=31;else IsGLES=false;
    check(ValidateGLESReplayCapabilities());check(queries==0 && errorReads==0);}
  // All 128 texture selectors and the separate generic buffer binding.
  for(unsigned i=0;i<128;++i){reset();HasExt[0]=false;auto s=uniform();s.TexBuffer[i].name=i+1;
    reject(s,"Replay initial state requires unsupported texture-buffer bindings");check(queries==0);}
  reset();HasExt[0]=false;auto s=uniform();s.BufferBindings[GLRenderState::eBufIdx_Texture].name=1;
  reject(s,"Replay initial state requires unsupported texture-buffer buffer binding");check(queries==0);
  reset();s.TexBuffer[127].name=42;check(s.ReplayCapabilityError()==nullptr && queries==0);
  // Every indexed blend field and every mask channel, not just enable/RGB.
  for(unsigned i=1;i<8;++i)for(unsigned field=0;field<7;++field){reset();HasExt[2]=HasExt[3]=false;s=uniform();blendChange(s.Blends[i],field);
    reject(s,"Replay initial state requires unsupported nonuniform indexed blend state");}
  for(unsigned i=1;i<8;++i)for(unsigned channel=0;channel<4;++channel){reset();HasExt[2]=HasExt[3]=false;s=uniform();maskChange(s.ColorMasks[i],channel);
    reject(s,"Replay initial state requires unsupported nonuniform indexed color masks");}
  for(unsigned flags=0;flags<4;++flags){reset();HasExt[2]=flags&1;HasExt[3]=flags&2;s=uniform();
    check(s.ReplayCapabilityError()==nullptr);maskChange(s.ColorMasks[7],0);
    if(flags)check(s.ReplayCapabilityError()==nullptr);else reject(s,"Replay initial state requires unsupported nonuniform indexed color masks");}
  // Canonical shorter capture prefix, unused tail, holes, and all-NONE clear.
  for(unsigned n=0;n<=8;++n){reset();HasExt[2]=HasExt[3]=false;s=uniform(n);check(s.ReplayCapabilityError()==nullptr);}
  for(unsigned n=1;n<8;++n)for(unsigned f=1;f<7;++f){reset();HasExt[2]=HasExt[3]=false;s=uniform(n);blendChange(s.Blends[n],f);
    reject(s,"Replay initial state has noncanonical unused indexed state");}
  for(unsigned n=0;n<8;++n)for(unsigned f=0;f<4;++f){reset();HasExt[2]=HasExt[3]=false;s=uniform(n);maskChange(s.ColorMasks[n],f);
    reject(s,"Replay initial state has noncanonical unused indexed state");}
  for(unsigned hole=0;hole<7;++hole){reset();HasExt[2]=HasExt[3]=false;s=uniform();s.Blends[hole]={};s.ColorMasks[hole]={};
    reject(s,"Replay initial state has an ambiguous indexed-state prefix");}
  for(int limit:{-1,0,9,2147483647}){reset();HasExt[2]=HasExt[3]=false;maxDraws=limit;s=uniform();
    reject(s,"Replay initial-state draw-buffer limit is unsupported");}
  for(int limit=1;limit<=8;++limit){reset();HasExt[2]=HasExt[3]=false;maxDraws=limit;s=uniform();check(s.ReplayCapabilityError()==nullptr);}
  for(GLenum error:{eGL_INVALID_ENUM,eGL_INVALID_OPERATION,eGL_OUT_OF_MEMORY}){reset();HasExt[2]=HasExt[3]=false;limitError=error;s=uniform();
    reject(s,"Replay initial-state draw-buffer limit is unsupported");}
  for(unsigned missing=0;missing<2;++missing){reset();HasExt[2]=HasExt[3]=false;s=uniform();
    if(missing)GL.glGetError=nullptr;else GL.glGetIntegerv=nullptr;
    reject(s,"Replay initial-state capability query functions are missing");}
  reset();HasExt[2]=HasExt[3]=false;pending.push_back(eGL_INVALID_OPERATION);s=uniform();
  reject(s,"Replay initial-state capability validation found a pending GL error");check(queries==0);
  reset();HasExt[0]=HasExt[2]=HasExt[3]=false;s=uniform();s.ContextPresent=false;s.TexBuffer[0].name=1;GL.glGetError=nullptr;
  check(s.ReplayCapabilityError()==nullptr && queries==0);
  // Actual ApplyState prefix must fail before mutation, preserve older fatal,
  // and retain ContextPresent/null-context early returns.
  reset();HasExt[2]=HasExt[3]=false;s=uniform();s.Blends[7].Enabled=true;
  WrappedOpenGL driver;s.ApplyState(&driver);check(driver.m_FatalError==ResultCode::APIHardwareUnsupported && driver.mutations==0);
  reset();driver={};driver.m_FatalError=ResultCode::ExistingFailure;s=uniform();s.ApplyState(&driver);
  check(driver.m_FatalError==ResultCode::ExistingFailure && driver.mutations==0 && queries==0);
  reset();driver={};s=uniform();s.ApplyState(&driver);check(driver.mutations==1 && driver.m_FatalError==ResultCode::Succeeded);
  reset();driver={};s.ContextPresent=false;s.ApplyState(&driver);check(driver.mutations==0 && queries==0);
  reset();driver={};driver.context.ctx=nullptr;s=uniform();s.ApplyState(&driver);check(driver.mutations==0 && queries==0);
  std::printf("PASS capability/state checks=%llu\n",static_cast<unsigned long long>(checks));
 }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
'''.replace("ACTUAL_CAPABILITY", capability).replace("ACTUAL_CHECKER", checker).replace("ACTUAL_APPLY", apply)
        with tempfile.TemporaryDirectory(prefix="snapshot-capability-guards-") as directory:
            path = Path(directory)
            cpp = path / "actual-capability-guards.cpp"
            cpp.write_text(program)
            modes = [("normal", ["-O2"])]
            if os.environ.get("PVRGPU_SNAPSHOT_TEST_SANITIZERS") == "1":
                modes.append(("sanitized", ["-O1", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]))
            for name, flags in modes:
                with self.subTest(mode=name):
                    executable = path / name
                    result = subprocess.run([*shlex.split(compiler), "-std=c++14", "-Wall", "-Wextra", "-Werror",
                        *flags, str(cpp), "-o", str(executable)], text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    result = subprocess.run([str(executable)], text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertRegex(result.stdout, r"^PASS capability/state checks=[1-9][0-9]*\n$")


class PrivateSnapshotBuilder(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="snapshot-builder-unit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source with spaces"
        self.base = self.root / "base with spaces"
        self.source.mkdir(); self.base.mkdir()
        self.old = {name: ("// baseline " + name + "\n").encode() for name in builder.CHANGED_EXISTING}
        self.new = {name: ("// snapshot " + name + "\n").encode()
                    for name in builder.CHANGED_EXISTING | builder.ADDED}
        self.old["vendor.txt"] = b"pinned existing vendor modification\n"
        self.old["empty.txt"] = b""
        for name in builder.V2_CHANGED_EXISTING - builder.CHANGED_EXISTING:
            self.old[name] = ("// baseline " + name + "\n").encode()
        self.extra_args = []
        for name, data in self.old.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Unit Test", "-c", "user.email=unit@example.invalid", "commit", "-qm", "fixture")
        head = self.git("rev-parse", "HEAD").strip()
        self.patch = self.root / "snapshot.patch"
        diffs = []
        for name in sorted(self.new):
            before = self.old.get(name, b"").decode().splitlines(keepends=True)
            after = self.new[name].decode().splitlines(keepends=True)
            diffs.append(f"diff --git a/{name} b/{name}\n")
            if name not in self.old:
                diffs.append("new file mode 100644\n")
            diffs.extend(difflib.unified_diff(before, after,
                fromfile="a/" + name if before else "/dev/null", tofile="b/" + name))
        self.patch.write_text("".join(diffs))
        self.lock_data = {"schema": builder.LOCK_SCHEMA, "source_git_head": head,
            "baseline_tracked_tree_sha256": builder.tree_digest({name: sha(data) for name, data in self.old.items()}),
            "patch_sha256": sha(self.patch.read_bytes()),
            "files": {name: {"base_sha256": sha(self.old[name]) if name in self.old else None,
                              "patched_sha256": sha(data)} for name, data in self.new.items()}}
        self.lock = self.root / "snapshot.lock.json"; self.write_lock()
        (self.base / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=" + str(self.source) + "\n")
        (self.base / "build.ninja").write_text("# mocked command inventory, never build\n")
        self.originals = {self.base / target: b"old object " + name.encode() for name, target in builder.TARGETS.items()}
        self.originals[self.base / "unchanged.o"] = b"reused unchanged object"
        self.originals[self.base / "lib/librenderdoc.dylib"] = b"installed library remains untouched"
        for path, data in self.originals.items():
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        self.compiler = self.root / "mock compiler"
        self.compiler.write_text("#!" + sys.executable + "\n" +
            "import json, pathlib, sys\n"
            "args=sys.argv[1:]\n"
            "if args == ['--version']: print('mock compiler: protocol testing only'); sys.exit(0)\n"
            "if '-DFAIL_TEST_COMMAND' in args: sys.exit(23)\n"
            "out=pathlib.Path(args[args.index('-o')+1]); out.write_text(json.dumps(args))\n"
            "if '-MF' in args: pathlib.Path(args[args.index('-MF')+1]).write_text('private dependency file')\n")
        self.compiler.chmod(0o700)
        self.ninja = self.root / "mock ninja"
        self.ninja.write_text("#!" + sys.executable + "\n" +
            "import json, pathlib, sys\n"
            "a=sys.argv[1:]\n"
            "assert len(a)==5 and a[0]=='-C' and a[2:4]==['-t','commands'], 'must only query Ninja'\n"
            "print(json.loads((pathlib.Path(a[1])/'commands.json').read_text())[a[4]])\n")
        self.ninja.chmod(0o700)
        self.commands = {}
        for name, target in builder.TARGETS.items():
            command = [str(self.compiler), "-O3", "-DNDEBUG", "-arch", "x86_64", "-I" + str(self.source),
                       "-MD", "-MT", target, "-MF", target + ".d", "-o", target, "-c", str(self.source / name)]
            self.commands[target] = shlex.join(command)
        self.link_target = "lib/librenderdoc.dylib"
        self.link = [str(self.compiler), "-O3", "-dynamiclib", "-o", self.link_target,
                     "-install_name", "@rpath/librenderdoc.dylib", *builder.TARGETS.values(), "unchanged.o"]
        self.set_link()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.source), *args], text=True)

    def write_lock(self):
        self.lock.write_text(json.dumps(self.lock_data))

    def set_link(self, suffix=""):
        # Non-target generator redirection is inventory only; it is never run.
        self.commands[self.link_target] = "printf unused > must-not-exist.c\n: && " + shlex.join(self.link) + " && :" + suffix
        self.write_commands()

    def write_commands(self):
        (self.base / "commands.json").write_text(json.dumps(self.commands))

    def run_build(self, expected=0, out=None):
        out = out or self.root / "private output"
        process = subprocess.run([sys.executable, str(SCRIPT / "build_renderdoc_snapshot.py"),
            str(self.source), str(self.base), str(out), "--lock", str(self.lock), "--patch", str(self.patch),
            "--ninja", str(self.ninja), "--jobs", "2", *self.extra_args], text=True, capture_output=True)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        for path, data in self.originals.items():
            self.assertEqual(path.read_bytes(), data, str(path))
        self.assertFalse((self.base / "must-not-exist.c").exists())
        self.assertFalse(any(self.base.rglob("*.o.d")))
        return out, process

    def assert_built(self, out):
        receipt = json.loads((out / "build-receipt.json").read_text())
        self.assertEqual(receipt["status"], "PASS")
        self.assertTrue(receipt["source_unchanged"])
        self.assertTrue(receipt["shared_outputs_unchanged"])
        self.assertEqual((out / "source.patch").read_bytes(), self.patch.read_bytes())
        self.assertEqual((out / "source-lock.json").read_bytes(), self.lock.read_bytes())
        self.assertEqual(len(receipt["builder_sources"]), 3)
        for name, data in self.new.items():
            self.assertEqual((out / "source" / name).read_bytes(), data)
        for args in receipt["compile_commands"].values():
            self.assertIn("-O3", args); self.assertIn("-DNDEBUG", args)
            self.assertIn("-I" + str(out / "source"), args)
            self.assertNotIn(str(self.source), "\n".join(args))
            for flag in ("-o", "-MF", "-MT"):
                self.assertTrue(Path(args[args.index(flag) + 1]).is_relative_to(out / "objects"))
        link = receipt["link_command"]
        self.assertEqual(link[link.index("-install_name") + 1], str(out / "lib/librenderdoc-drawlist-snapshot.dylib"))
        self.assertEqual(len(receipt["object_identities"]), len(builder.targets_for(self.lock_data)))
        self.assertEqual(receipt["library_identity"]["sha256"], sha(Path(receipt["library"]).read_bytes()))

    def test_exact_baseline_build_is_private(self):
        out, _ = self.run_build()
        self.assert_built(out)
        for name, data in self.old.items():
            self.assertEqual((self.source / name).read_bytes(), data)
        self.assertFalse(any((self.source / name).exists() for name in builder.ADDED))

    def test_patch_under_repository_subdirectory_has_complete_inventory(self):
        directory = self.source / ".git" / "patches"
        directory.mkdir()
        patch = directory / "snapshot.patch"; patch.write_bytes(self.patch.read_bytes())
        self.assertEqual(builder.patch_names(patch, "git"), set(self.new))

    def test_output_under_another_worktree_uses_private_git_root(self):
        unrelated = self.root / "unrelated-worktree"; unrelated.mkdir()
        subprocess.check_call(["git", "init", "-q", str(unrelated)])
        out, _ = self.run_build(out=unrelated / "private-build")
        self.assert_built(out)

    def test_already_patched_and_mixed_sources_normalize(self):
        for mode in ("mixed", "patched"):
            with self.subTest(mode=mode):
                for i, (name, data) in enumerate(sorted(self.new.items())):
                    if mode == "patched" or i % 2 == 0:
                        path = self.source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
                out, _ = self.run_build(out=self.root / mode)
                self.assert_built(out)

    def test_unknown_vendor_edit_refuses(self):
        (self.source / "vendor.txt").write_text("unreviewed vendor changes\n")
        _, result = self.run_build(expected=1)
        self.assertIn("pinned Mesa/vendor baseline", result.stderr)

    def test_partial_patch_refuses(self):
        (self.source / next(iter(builder.CHANGED_EXISTING))).write_text("half applied\n")
        _, result = self.run_build(expected=1)
        self.assertIn("unrecognized source bytes", result.stderr)

    def test_unknown_untracked_file_refuses(self):
        (self.source / "unexpected.cpp").write_text("untracked\n")
        _, result = self.run_build(expected=1)
        self.assertIn("unknown untracked", result.stderr)

    def test_patch_hash_and_head_refuse(self):
        self.lock_data["patch_sha256"] = "0" * 64; self.write_lock()
        _, result = self.run_build(expected=1)
        self.assertIn("patch hash mismatch", result.stderr)
        self.lock_data["patch_sha256"] = sha(self.patch.read_bytes())
        self.lock_data["source_git_head"] = "0" * 40; self.write_lock()
        _, result = self.run_build(expected=1)
        self.assertIn("Git HEAD", result.stderr)

    def test_wrong_build_source_refuses(self):
        (self.base / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=" + str(self.root) + "\n")
        _, result = self.run_build(expected=1)
        self.assertIn("different source", result.stderr)

    def test_fresh_and_external_output_required(self):
        occupied = self.root / "occupied"; occupied.mkdir()
        for path in (occupied, self.base / "private", self.source / "private"):
            with self.subTest(path=path):
                self.run_build(expected=1, out=path)

    def test_source_and_output_symlinks_refuse(self):
        vendor = self.source / "vendor.txt"
        elsewhere = self.root / "vendor-copy"; elsewhere.write_bytes(vendor.read_bytes())
        vendor.unlink(); vendor.symlink_to(elsewhere)
        _, result = self.run_build(expected=1)
        self.assertIn("symlink", result.stderr.lower())
        vendor.unlink(); vendor.write_bytes(self.old["vendor.txt"])
        link = self.root / "output-link"; link.symlink_to(self.root / "uncreated")
        self.run_build(expected=1, out=link)

    def test_extra_link_step_and_missing_replacement_refuse(self):
        self.set_link(" && touch forbidden")
        _, result = self.run_build(expected=1)
        self.assertIn("unsupported prerequisite/post-link", result.stderr)
        self.link.remove(next(iter(builder.TARGETS.values()))); self.set_link()
        _, result = self.run_build(expected=1)
        self.assertIn("all changed objects", result.stderr)

    def test_auxiliary_compiler_outputs_refuse(self):
        target = next(iter(builder.TARGETS.values()))
        self.commands[target] += " -ftime-trace"
        self.write_commands()
        _, result = self.run_build(expected=1)
        self.assertIn("auxiliary output flags", result.stderr)

    def test_response_file_refuses(self):
        self.link.append("@uninspected-args.rsp"); self.set_link()
        _, result = self.run_build(expected=1)
        self.assertIn("response files", result.stderr)

    def test_failed_compile_retains_failure_receipt(self):
        target = next(iter(builder.TARGETS.values()))
        self.commands[target] += " -DFAIL_TEST_COMMAND"; self.write_commands()
        out, _ = self.run_build(expected=1)
        receipt = json.loads((out / "build-receipt.json").read_text())
        self.assertEqual(receipt["status"], "FAIL")
        self.assertTrue(receipt["source_unchanged"] and receipt["shared_outputs_unchanged"])
        self.assertNotIn("library_identity", receipt)

    def test_relative_path_contract(self):
        for value in ("../escape", "/absolute", "a/../b", "a//b", "a\nb", "a\0b", "./a"):
            with self.subTest(value=value), self.assertRaises(builder.ReplayError):
                builder.relative_name(value)

    def configure_v2(self, source_mode="legacy"):
        legacy_patch = self.root / "v1.patch"
        legacy_patch.write_bytes(self.patch.read_bytes())
        legacy_lock = self.root / "v1.lock.json"
        legacy_lock.write_bytes(self.lock.read_bytes())
        self.legacy_files = dict(self.new)
        self.extra_args = ["--legacy-lock", str(legacy_lock), "--legacy-patch", str(legacy_patch)]
        if source_mode == "legacy":
            for name, data in self.legacy_files.items():
                path = self.source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        self.new = {name: data.replace(b"// snapshot ", b"// snapshot v2 ")
                    for name, data in self.new.items()}
        for name in builder.V2_CHANGED_EXISTING - builder.CHANGED_EXISTING:
            self.new[name] = ("// snapshot v2 " + name + "\n").encode()
        diffs = []
        for name in sorted(self.new):
            before = self.old.get(name, b"").decode().splitlines(keepends=True)
            after = self.new[name].decode().splitlines(keepends=True)
            diffs.append(f"diff --git a/{name} b/{name}\n")
            if not before:
                diffs.append("new file mode 100644\n")
            diffs.extend(difflib.unified_diff(before, after,
                fromfile="a/" + name if before else "/dev/null", tofile="b/" + name))
        self.patch.write_text("".join(diffs))
        self.lock_data.update(schema=builder.V2_LOCK_SCHEMA,
            patch_sha256=sha(self.patch.read_bytes()),
            files={name: {"base_sha256": sha(self.old[name]) if name in self.old else None,
                          "patched_sha256": sha(data)} for name, data in self.new.items()})
        self.write_lock()
        for name in set(builder.V2_TARGETS) - set(builder.TARGETS):
            target = builder.V2_TARGETS[name]
            self.originals[self.base / target] = ("old object remains unchanged: " + name).encode()
            (self.base / target).parent.mkdir(parents=True, exist_ok=True)
            (self.base / target).write_bytes(self.originals[self.base / target])
            self.commands[target] = shlex.join([str(self.compiler), "-O3", "-DNDEBUG", "-arch", "x86_64",
                "-I" + str(self.source), "-MD", "-MT", target, "-MF", target + ".d", "-o", target,
                "-c", str(self.source / name)])
            self.link.append(target)
        self.set_link()

    def test_v2_normalizes_legacy_source_with_the_exact_legacy_patch(self):
        self.configure_v2()
        out, _ = self.run_build()
        self.assert_built(out)
        for name, data in self.legacy_files.items():
            self.assertEqual((self.source / name).read_bytes(), data)
        receipt = json.loads((out / "build-receipt.json").read_text())
        self.assertEqual(len(receipt["legacy_inputs"]), 2)
        self.assertEqual(len(receipt["private_legacy_inputs"]), 2)
        for identity in receipt["private_legacy_inputs"]:
            self.assertEqual(sha(Path(identity["path"]).read_bytes()), identity["sha256"])

    def test_v2_baseline_and_mixed_exact_inputs(self):
        self.configure_v2(source_mode="base")
        out, _ = self.run_build(out=self.root / "v2-clean")
        self.assert_built(out)
        for i, (name, data) in enumerate(sorted(self.new.items())):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.legacy_files.get(name, data) if i % 2 else data)
        out, _ = self.run_build(out=self.root / "v2-mixed")
        self.assert_built(out)

    def test_v2_rejects_legacy_lock_from_a_different_baseline(self):
        self.configure_v2()
        path = Path(self.extra_args[1])
        value = json.loads(path.read_text())
        value["baseline_tracked_tree_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        _, result = self.run_build(expected=1)
        self.assertIn("same v1 baseline", result.stderr)

    def test_v2_rejects_tampered_legacy_patch(self):
        self.configure_v2()
        Path(self.extra_args[3]).write_text("untrusted replacement patch\n")
        _, result = self.run_build(expected=1)
        self.assertIn("patch hash mismatch", result.stderr)

    def test_v2_requires_all_eight_changed_objects_in_link(self):
        self.configure_v2()
        self.assertEqual(len(builder.targets_for(self.lock_data)), 8)
        self.assertEqual(len(builder.existing_for(self.lock_data)), 13)
        for index, name in enumerate(sorted(set(builder.V2_TARGETS) - set(builder.TARGETS))):
            with self.subTest(name=name):
                target = builder.V2_TARGETS[name]
                self.link.remove(target)
                self.set_link()
                _, result = self.run_build(out=self.root / f"missing-v2-object-{index}", expected=1)
                self.assertIn("all changed objects", result.stderr)
                self.link.append(target)

    def test_v2_rejects_arbitrary_partial_source_edits(self):
        self.configure_v2()
        (self.source / "renderdoc/driver/gl/gl_driver.cpp").write_text("half v1 half v2\n")
        _, result = self.run_build(expected=1)
        self.assertIn("unrecognized source bytes", result.stderr)


if __name__ == "__main__":
    unittest.main()
