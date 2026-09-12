// SPDX-License-Identifier: MIT
// API32 deep-copy and fail-closed CubeArray transport, using genuine compiler
// VS/FS bytes and the full native model. Inputs are independently colored cubes.
#include "pvrgpu_systemc_api.h"
#include "pco_cube_array_fixtures.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
unsigned checks=0;
void Check(bool b,const std::string&w){++checks;if(!b)throw std::runtime_error(w);}
uint32_t Bits(float f){uint32_t b;std::memcpy(&b,&f,4);return b;}
struct Fixture {
  std::vector<uint8_t> vs=pvrgpu::stub::test::CubeArrayVertexFixture();
  std::vector<uint8_t> fs=pvrgpu::stub::test::CubeArray601Fixture();
  std::array<float,6> vertices{-1,-1,3,-1,-1,3};
  std::array<uint8_t,18*4> bytes{};
  std::array<uint32_t,28> shared{};
  std::array<char,40> format{};
  pvrgpu_systemc_varying_binding empty_binding{};
  pvrgpu_systemc_pco_sequence_texture texture{};
  pvrgpu_systemc_driver_command draw{},logical{};
  Fixture(){
    std::strcpy(format.data(),"PIPE_FORMAT_R8G8B8A8_UNORM");
    for(unsigned face=0;face<18;++face)for(unsigned c=0;c<4;++c)
      bytes[face*4+c]=uint8_t(5+face/6*30+face%6*3+c);
    const uint64_t w0=1ULL|(3ULL<<5)|(2ULL<<8)|(1ULL<<11)|(12ULL<<27);
    shared[0]=uint32_t(w0);shared[1]=w0>>32;shared[2]=1|(2<<4);shared[4]=4;
    shared[8]=0xfff;shared[16]=0xfff;shared[17]=(1U<<4)|(1U<<6);
    shared[20]=Bits(1);shared[23]=Bits(1);shared[24]=Bits(0);
    texture.source=PVRGPU_SYSTEMC_PCO_TEXTURE_EXTERNAL_PAYLOAD;
    texture.stage=PVRGPU_SYSTEMC_PCO_SHADER_STAGE_FRAGMENT;texture.format=format.data();
    texture.bytes=bytes.data();texture.bytes_size=texture.declared_bytes_size=bytes.size();
    texture.mip_count=1;texture.mip[0]={1,1,4,0};texture.normalized_coordinates=1;
    texture.wrap_u=texture.wrap_v=PVRGPU_SYSTEMC_PCO_TEXTURE_WRAP_REPEAT;
    texture.texture_kind=4;texture.layers=18;texture.sample_count=1;
    draw.version=PVRGPU_SYSTEMC_API_VERSION;draw.command="draw_pco_triangles";
    draw.case_name="native.cube-array-api";draw.format="PIPE_FORMAT_R8G8B8A8_UNORM";draw.frame=1;
    draw.width=draw.height=draw.framebuffer_width=draw.framebuffer_height=4;
    draw.raw_vertex_data=reinterpret_cast<const uint8_t*>(vertices.data());draw.raw_vertex_data_size=sizeof(vertices);
    draw.vertex_stride=8;draw.vertex_count=3;draw.instance_count=1;draw.primitive_mode=4;draw.render_target_count=1;
    draw.vertex_attribute_count=1;draw.vertex_attribute_components[0]=2;
    draw.vertex_pco=vs.data();draw.vertex_pco_size=vs.size();draw.fragment_pco=fs.data();draw.fragment_pco_size=fs.size();
    draw.vertex_pco_abi={4,4,4,0,0,0,0,0,0,0};draw.fragment_pco_abi={20,0,0,4,28,20,8,0,0,0};
    draw.fragment_shared=shared.data();draw.fragment_shared_count=shared.size();draw.sampled_texture_count=1;
    draw.position_output_count=draw.fragment_position_count=4;
    draw.varying_output_start=draw.fragment_varying_start=4;draw.varying_bindings=&empty_binding;
    draw.fragment_output_mask[0]=15;
    const std::array<uint32_t,3> viewport{Bits(2),Bits(2),Bits(.5F)};
    std::copy(viewport.begin(),viewport.end(),draw.viewport_scale_bits);std::copy(viewport.begin(),viewport.end(),draw.viewport_translate_bits);
    draw.half_pixel_center=draw.depth_clip_near=draw.depth_clip_far=1;
    draw.sample_mask=UINT32_MAX;draw.color_mask=15;
    draw.blend_source_rgb_factor=draw.blend_source_alpha_factor=PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
    draw.color_attachment_source_command_index=draw.depth_attachment_source_command_index=PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
    logical.version=PVRGPU_SYSTEMC_API_VERSION;logical.command="draw_pco_sequence";
    logical.case_name=draw.case_name;logical.format=draw.format;logical.frame=1;
    logical.width=logical.height=logical.framebuffer_width=logical.framebuffer_height=4;
    logical.render_target_count=1;logical.draw_count=1;logical.ia_vertices=3;logical.ia_primitives=logical.clip_invocations=1;
    logical.pco_sequence_command_count=1;logical.pco_sequence_commands=&draw;
    logical.pco_sequence_texture_count=1;logical.pco_sequence_textures=&texture;
  }
};
struct Submission {
  std::string json,err,out;pvrgpu_systemc_submit_info info{};
  Submission(const std::filesystem::path&root,Fixture &f){
    std::filesystem::create_directories(root);json=(root/"model.jsonl").string();err=(root/"stderr.log").string();out=(root/"model").string();
    info.version=PVRGPU_SYSTEMC_API_VERSION;info.command=&f.logical;info.jsonl_path=json.c_str();info.stderr_path=err.c_str();info.outdir=out.c_str();info.memory_mode="direct";
  }
};
void Negatives(const std::filesystem::path&root){
  for(unsigned n=0;n<22;++n){Fixture f;const char *why="cube array";
    std::array<pvrgpu_systemc_pco_sequence_texture,2> pair{f.texture,f.texture};
    switch(n){
      case 0:f.texture.layers=0;why="whole-cube";break;
      case 1:f.texture.layers=17;why="whole-cube";break;
      case 2:f.texture.layers=12294;why="whole-cube";break;
      case 3:f.texture.sample_count=4;why="whole-cube";break;
      case 4:f.texture.stage=PVRGPU_SYSTEMC_PCO_SHADER_STAGE_VERTEX;why="whole-cube";break;
      case 5:f.texture.source=PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_COLOR_ATTACHMENT;why="whole-cube";break;
      case 6:f.texture.mip[0].height=2;break;
      case 7:f.texture.mip[0].row_pitch=8;break;
      case 8:f.texture.format="PIPE_FORMAT_ASTC_4x4";why="whole-cube";break;
      case 9:f.texture.bytes_size--;why="payload";break;
      case 10:f.texture.declared_bytes_size--;why="mip";break;
      case 11:f.shared[2]+=1<<4;break;
      case 12:f.shared[4]*=6;break;
      case 13:f.shared[0]=(f.shared[0]&~7U)|4U;break;
      case 14:f.shared[7]=0x100;break;
      case 15:f.shared[12]=3;break;
      case 16:f.texture.descriptor_set=12;why="descriptor set";break;
      case 17:f.logical.version=31;why="version";break;
      case 18:f.draw.version=31;why="version";break;
      case 19:f.texture.descriptor_set=1;why="stage-dense";break;
      case 20:f.logical.pco_sequence_textures=pair.data();f.logical.pco_sequence_texture_count=2;f.draw.sampled_texture_count=2;why="duplicated";break;
      case 21:f.draw.sampled_texture_count=2;why="truncated";break;
    }
    Submission s(root/("negative-"+std::to_string(n)),f);std::array<char,2048>error{};
    Check(pvrgpu_systemc_submit_driver_command(&s.info,error.data(),error.size())==2,"malformed API CubeArray refused "+std::to_string(n));
    Check(std::string(error.data()).find(why)!=std::string::npos,"precise refusal "+std::to_string(n)+": "+error.data());
  }
}
void Native(const std::filesystem::path&root){
  Fixture f;Submission s(root/"native-deep-copy",f);std::array<char,2048>error{};
  Check(pvrgpu_systemc_submit_driver_command(&s.info,error.data(),error.size())==0,std::string("native CubeArray submit: ")+error.data());
  // The deferred native job must own the real input data, not these pointers.
  f.bytes.fill(0xee);f.shared.fill(0);f.vertices.fill(0);
  std::fill(f.vs.begin(),f.vs.end(),0);std::fill(f.fs.begin(),f.fs.end(),0);
  std::strcpy(f.format.data(),"MUTATED");f.texture.layers=6;f.texture.texture_kind=0;
  std::array<uint8_t,64> pixels{};pixels.fill(0xa5);
  pvrgpu_systemc_readback_info r{};r.version=PVRGPU_SYSTEMC_API_VERSION;r.width=r.height=4;
  r.bytes_per_pixel=4;r.pixels=pixels.data();r.pixels_size=pixels.size();
  Check(pvrgpu_systemc_flush_readback(&r,error.data(),error.size())==0&&r.pixels_written==1,std::string("native CubeArray completion: ")+error.data());
  for(unsigned pixel=0;pixel<16;++pixel)for(unsigned c=0;c<4;++c)
    Check(pixels[pixel*4+c]==35+c,"native output must sample cube1 positive-X, not cube0 or mutated input");
}
void NativeTwelve(const std::filesystem::path&root){
  Fixture f;f.fs=pvrgpu::stub::test::TwelveTextureFixture();
  std::array<uint32_t,256> shared{};
  std::array<pvrgpu_systemc_pco_sequence_texture,12> textures{};
  std::array<std::array<uint8_t,72>,12> bytes{};
  std::array<uint8_t,32> ubo_bytes{};
  std::array<unsigned,4> expected{};
  for(unsigned slot=0;slot<12;++slot){
    std::copy_n(f.shared.begin(),20,shared.begin()+slot*20);
    auto&t=textures[slot];t=f.texture;t.descriptor_set=slot;
    t.texture_kind=slot==6?4:((slot==7||slot==8)?1:0);
    t.layers=slot==6?18:((slot==7||slot==8)?2:1);
    t.bytes=bytes[slot].data();t.bytes_size=t.declared_bytes_size=t.layers*4;
    shared[slot*20+2]=1|((slot==6?2:t.layers-1)<<4);
    if(t.texture_kind==0){
      shared[slot*20]=(shared[slot*20]&~7U)|4U;
      shared[slot*20+2]=0;shared[slot*20+3]=1U<<28;
    }
    for(unsigned c=0;c<4;++c){
      const unsigned value=1+(slot+c)%3;expected[c]+=(slot+1)*value;
      for(unsigned face=0;face<t.layers;++face)bytes[slot][face*4+c]=uint8_t(value);
    }
  }
  shared[242]=32;
  shared[244]=Bits(.25F);shared[245]=Bits(.75F);shared[246]=Bits(.5F);shared[247]=Bits(1);
  shared[252]=Bits(0);shared[253]=Bits(1);shared[254]=Bits(2);
  pvrgpu_systemc_pco_uniform_buffer ubo{PVRGPU_SYSTEMC_PCO_SHADER_STAGE_FRAGMENT,0,ubo_bytes.data(),ubo_bytes.size()};
  f.draw.fragment_pco=f.fs.data();f.draw.fragment_pco_size=f.fs.size();
  f.draw.fragment_pco_abi={64,0,0,4,256,244,12,0,240,1};
  f.draw.fragment_shared=shared.data();f.draw.fragment_shared_count=shared.size();
  f.draw.sampled_texture_count=12;f.draw.uniform_buffers=&ubo;f.draw.uniform_buffer_count=1;
  f.logical.pco_sequence_textures=textures.data();f.logical.pco_sequence_texture_count=textures.size();
  Submission s(root/"native-twelve-deep-copy",f);std::array<char,2048>error{};
  Check(pvrgpu_systemc_submit_driver_command(&s.info,error.data(),error.size())==0,std::string("native twelve-slot submit: ")+error.data());
  for(auto&b:bytes)b.fill(0xee);shared.fill(0);ubo_bytes.fill(0xff);
  for(auto&t:textures){t.layers=0;t.descriptor_set=12;}
  std::fill(f.fs.begin(),f.fs.end(),0);std::strcpy(f.format.data(),"MUTATED");
  std::array<uint8_t,64>pixels{};
  pvrgpu_systemc_readback_info r{};r.version=PVRGPU_SYSTEMC_API_VERSION;r.width=r.height=4;
  r.bytes_per_pixel=4;r.pixels=pixels.data();r.pixels_size=pixels.size();
  Check(pvrgpu_systemc_flush_readback(&r,error.data(),error.size())==0&&r.pixels_written==1,std::string("native twelve-slot completion: ")+error.data());
  for(unsigned pixel=0;pixel<16;++pixel)for(unsigned c=0;c<4;++c)
    Check(expected[c]<255&&pixels[pixel*4+c]==expected[c],"native twelve-slot exact weighted UNORM sum from all real payloads");
}
}
int main(){try{
  static_assert(PVRGPU_SYSTEMC_API_VERSION==35&&PVRGPU_SYSTEMC_MAX_PCO_TEXTURES_PER_STAGE==16);
  static_assert(sizeof(void*)!=8||sizeof(pvrgpu_systemc_pco_sequence_texture)==352,"API32 retains texture struct size");
  const auto root=std::filesystem::temp_directory_path()/("pvrgpu-cube-array-api-"+std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
  Negatives(root);Native(root);NativeTwelve(root);std::cout<<"CubeArray API32: "<<checks<<" checks PASS; artifacts="<<root<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<"CubeArray API after "<<checks<<" checks: "<<e.what()<<'\n';return 1;}}
