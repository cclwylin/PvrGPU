// API31: actual deferred native LOAD/draw/alias/readback with owned format
// strings. Reuses unmodified genuine PCO fixture binaries, no reference image.
#include "pvrgpu_systemc_api.h"
#include "shader/pco_iss.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
unsigned checks = 0;
constexpr std::array<const char *,3> kFormats={"PIPE_FORMAT_R8G8B8A8_UNORM",
  "PIPE_FORMAT_R10G10B10A2_UNORM","PIPE_FORMAT_B10G10R10A2_UNORM"};
void Check(bool ok,const std::string &why) {++checks;if(!ok)throw std::runtime_error(why);}
struct Fixture {
  std::vector<std::uint8_t> vs=pvrgpu::stub::VaryingsOneVertexPcoBinary();
  std::vector<std::uint8_t> fs=pvrgpu::stub::VaryingsOneFragmentPcoBinary();
  std::array<float,12> vertices={-1,-1,0,1,1,-1,0,1,-1,1,0,1};
  std::vector<std::uint8_t> initial=std::vector<std::uint8_t>(4*4*4*4);
  std::array<std::array<char,48>,4> owned_names{};
  std::array<pvrgpu_systemc_driver_command,2> draws{};
  pvrgpu_systemc_driver_command logical{};
  explicit Fixture(unsigned first=0) {
    for(unsigned t=0;t<4;++t)std::strcpy(owned_names[t].data(),kFormats[(first+t)%3]);
    for(unsigned i=0;i<initial.size();++i)initial[i]=static_cast<std::uint8_t>(i*37+19);
    auto &d=draws[0];
    d.version=PVRGPU_SYSTEMC_API_VERSION;d.command="draw_pco_triangles";
    d.case_name="native.color-format-api";d.format=owned_names[0].data();d.frame=1;
    d.width=d.height=d.framebuffer_width=d.framebuffer_height=4;
    d.raw_vertex_data=reinterpret_cast<const std::uint8_t*>(vertices.data());d.raw_vertex_data_size=sizeof(vertices);
    d.vertex_stride=16;d.vertex_count=3;d.instance_count=1;d.primitive_mode=4;d.render_target_count=4;
    d.vertex_attribute_count=1;d.vertex_attribute_components[0]=4;
    d.vertex_pco=vs.data();d.vertex_pco_size=vs.size();d.fragment_pco=fs.data();d.fragment_pco_size=fs.size();
    d.vertex_pco_abi={4,4,8,0,0,0,0,0,0,0};d.fragment_pco_abi={4,0,0,20,0,0,0,0,0,0};
    d.position_output_count=4;d.varying_output_start=4;d.varying_output_count=4;
    d.fragment_position_count=4;d.fragment_varying_start=4;d.fragment_varying_count=16;
    d.fragment_output_mask[0]=15;
    const std::array<float,3> vp={2,2,.5f};
    std::memcpy(d.viewport_scale_bits,vp.data(),12);std::memcpy(d.viewport_translate_bits,vp.data(),12);
    d.half_pixel_center=d.depth_clip_near=d.depth_clip_far=1;d.sample_mask=UINT32_MAX;d.color_mask=15;
    d.blend_source_rgb_factor=d.blend_source_alpha_factor=PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
    d.color_attachment_source_command_index=d.depth_attachment_source_command_index=PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
    d.initial_color_attachment_bytes=initial.data();d.initial_color_attachment_bytes_size=initial.size();
    d.color_attachment_format_count=4;
    for(unsigned t=0;t<4;++t)d.color_attachment_formats[t]=owned_names[t].data();
    draws[1]=d;draws[1].initial_color_attachment_bytes=nullptr;draws[1].initial_color_attachment_bytes_size=0;
    draws[1].color_attachment_source_command_index=0;
    logical.version=PVRGPU_SYSTEMC_API_VERSION;logical.command="draw_pco_sequence";
    logical.case_name=d.case_name;logical.format=d.format;logical.frame=1;
    logical.width=logical.height=logical.framebuffer_width=logical.framebuffer_height=4;
    logical.render_target_count=4;logical.draw_count=2;logical.ia_vertices=6;logical.ia_primitives=logical.clip_invocations=2;
    logical.pco_sequence_command_count=2;logical.pco_sequence_commands=draws.data();
    logical.color_attachment_format_count=4;
    for(unsigned t=0;t<4;++t)logical.color_attachment_formats[t]=d.color_attachment_formats[t];
  }
};
struct Submission {
  std::string json,err,out; pvrgpu_systemc_submit_info info{};
  Submission(const std::filesystem::path &root,const pvrgpu_systemc_driver_command *d) {
    std::filesystem::create_directories(root);json=(root/"model.jsonl").string();err=(root/"stderr.log").string();out=(root/"output").string();
    info.version=PVRGPU_SYSTEMC_API_VERSION;info.command=d;info.jsonl_path=json.c_str();info.stderr_path=err.c_str();info.outdir=out.c_str();info.memory_mode="direct";
  }
};
void Reject(const std::filesystem::path &root,Fixture &f,const char *name,const char *expected) {
  Submission s(root/name,&f.logical);std::array<char,1024> error{};
  Check(pvrgpu_systemc_submit_driver_command(&s.info,error.data(),error.size())==2,std::string(name)+" rejected");
  Check(std::string(error.data()).find(expected)!=std::string::npos,std::string(name)+" reason: "+error.data());
}
void VerifyOldSize(const std::filesystem::path &root) {
#if !defined(_WIN32)
  static_assert(PVRGPU_SYSTEMC_API_VERSION==32);
  const auto page=static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  // API30 ends immediately before the new count, rounded to the old struct
  // alignment. Only the version may be read before refusing this command.
  const auto oldsize=(offsetof(pvrgpu_systemc_driver_command,color_attachment_format_count)+alignof(pvrgpu_systemc_driver_command)-1)&~(alignof(pvrgpu_systemc_driver_command)-1);
  Check(oldsize<page,"old command fits guard page");
  auto *base=static_cast<std::uint8_t*>(mmap(nullptr,page*2,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0));
  Check(base!=MAP_FAILED&&mprotect(base,page,PROT_READ|PROT_WRITE)==0,"guard pages");
  auto *old=base+page-oldsize;std::memset(old,0,oldsize);const std::uint32_t version=30;std::memcpy(old,&version,4);
  Check(mprotect(base,page,PROT_READ)==0,"guard readonly");
  auto *command=reinterpret_cast<const pvrgpu_systemc_driver_command*>(old);
  Submission s(root/"oldsize",command);std::array<char,512> error{};
  Check(pvrgpu_systemc_submit_driver_command(&s.info,error.data(),error.size())==2,"oldsize top reject");
  Check(std::string(error.data()).find("version")!=std::string::npos,"oldsize top reason");
  Fixture f;f.logical.pco_sequence_commands=command;f.logical.pco_sequence_command_count=1;s.info.command=&f.logical;
  Check(pvrgpu_systemc_submit_driver_command(&s.info,error.data(),error.size())==2,"oldsize nested reject");
  Check(std::string(error.data()).find("version")!=std::string::npos,"oldsize nested reason");
  Check(munmap(base,page*2)==0,"guard unmap");
#else
  (void)root;
#endif
}
void VerifyDepthReadbackTail() {
  std::array<std::uint8_t,4> pixels{};pixels.fill(0xa5);
  pvrgpu_systemc_readback_info r{};r.version=PVRGPU_SYSTEMC_API_VERSION;
  r.width=r.height=1;r.bytes_per_pixel=4;r.pixels=pixels.data();r.pixels_size=pixels.size();
  r.attachment=UINT32_MAX;r.depth_format=276;
  std::array<char,512> error{};
  for(const char *format:{kFormats[0],""}) {
    r.color_format=format;r.pixels_written=1;
    Check(pvrgpu_systemc_flush_readback(&r,error.data(),error.size())==2,"depth non-null color format rejected");
    Check(std::string(error.data()).find("requires a null color format")!=std::string::npos,"depth format refusal reason");
    Check(!r.pixels_written&&std::all_of(pixels.begin(),pixels.end(),[](auto b){return b==0xa5;}),"depth malformed tail cannot publish");
  }
}
void VerifyRun(const std::filesystem::path &root,unsigned first,bool legacy=false) {
  Fixture f(first);
  if(legacy) {
    for(auto &name:f.owned_names)std::strcpy(name.data(),kFormats[first]);
    f.logical.color_attachment_format_count=0;
    for(auto &p:f.logical.color_attachment_formats)p=nullptr;
    for(auto &d:f.draws){d.color_attachment_format_count=0;for(auto &p:d.color_attachment_formats)p=nullptr;}
  }
  const auto expected=f.initial;
  std::array<std::string,4> names;
  for(unsigned t=0;t<4;++t)names[t]=f.owned_names[t].data();
  Submission s(root/(std::string(legacy?"legacy-":"explicit-")+std::to_string(first)),&f.logical);
  std::array<char,1024> error{};
  Check(pvrgpu_systemc_submit_driver_command(&s.info,error.data(),error.size())==0,std::string("valid submit ")+error.data());
  for(auto &name:f.owned_names)std::strcpy(name.data(),"MUTATED_AFTER_SUBMIT");
  std::fill(f.initial.begin(),f.initial.end(),0xcc);
  std::array<std::uint8_t,64> pixels{};pixels.fill(0xa5);
  pvrgpu_systemc_readback_info r{};r.version=PVRGPU_SYSTEMC_API_VERSION;r.width=r.height=4;
  r.bytes_per_pixel=4;r.pixels=pixels.data();r.pixels_size=pixels.size();
  if(!legacy) {
    Check(pvrgpu_systemc_flush_readback(&r,error.data(),error.size())==2,"explicit missing readback format reject");
    Check(!r.pixels_written&&std::all_of(pixels.begin(),pixels.end(),[](auto b){return b==0xa5;}),"missing format cannot copy bytes");
  }
  for(unsigned target=0;target<4;++target) {
    r.attachment=target;r.color_format="PIPE_FORMAT_R32_UINT";
    Check(pvrgpu_systemc_flush_readback(&r,error.data(),error.size())==2,"wrong same4B readback format reject");
    Check(!r.pixels_written,"wrong format no publication");
    r.color_format=names[target].c_str();
    Check(pvrgpu_systemc_flush_readback(&r,error.data(),error.size())==0&&r.pixels_written==1,std::string("exact target readback ")+error.data());
    for(unsigned pixel=0;pixel<16;++pixel) {
      if(target || pixel%4+pixel/4>3)
        Check(!std::memcmp(pixels.data()+pixel*4,expected.data()+target*64+pixel*4,4),"target-major raw LOAD retained");
    }
    if(!target) {
      const std::uint32_t shaded=first ? 0xc0000000U : 0xff000000U;
      Check(!std::memcmp(pixels.data(),&shaded,4),"real shaded target0 correct native packing");
    }
    if(legacy){r.color_format=nullptr;Check(pvrgpu_systemc_flush_readback(&r,error.data(),error.size())==0&&r.pixels_written,"legacy noformat readback");}
  }
}
}
int main() {
  try {
    const auto root=std::filesystem::temp_directory_path()/("pvrgpu-color-formats-"+std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    VerifyOldSize(root);
    VerifyDepthReadbackTail();
    {Fixture f;f.draws[0].color_attachment_format_count=3;Reject(root,f,"count","color attachment formats");}
    {Fixture f;f.draws[0].color_attachment_formats[1]=nullptr;Reject(root,f,"null","color attachment formats");}
    {Fixture f;f.draws[0].color_attachment_formats[1]="";Reject(root,f,"empty","color attachment formats");}
    {Fixture f;f.draws[0].color_attachment_formats[1]="PIPE_FORMAT_R32_UINT";Reject(root,f,"integer","color attachment formats");}
    {Fixture f;f.draws[0].color_attachment_formats[0]=kFormats[1];Reject(root,f,"target0","color attachment formats");}
    {Fixture f;f.draws[0].color_attachment_format_count=0;Reject(root,f,"legacy-nonnull","inactive entry");}
    {Fixture f;f.draws[1].color_attachment_formats[1]=kFormats[0];Reject(root,f,"alias-format-swap","alias format/extent mismatch");}
    {Fixture f;--f.draws[0].initial_color_attachment_bytes_size;Reject(root,f,"truncated-load","byte count");}
    {Fixture f;++f.draws[0].initial_color_attachment_bytes_size;Reject(root,f,"oversized-load","byte count");}
    for(unsigned first=0;first<3;++first){VerifyRun(root,first);VerifyRun(root,first,true);}
    std::cout<<"color attachment API31: PASS "<<checks<<" checks; artifacts="<<root<<'\n';
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
  return 0;
}
