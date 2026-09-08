// SPDX-License-Identifier: MIT
// Real compiled VS -> independent GS -> layered ISP/PBE -> DRAM readback.
#include "pvrgpu_systemc_api.h"
#include "pco_geometry_compiler_fixtures.h"
#include "shader/pco_iss.h"
#include "model_types.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {
unsigned checks = 0;
void Check(bool value, const std::string &message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}
void Run(const std::filesystem::path &root, const char *mode, unsigned layers, bool depth) {
  auto vs = pvrgpu::stub::ConditionalsVertexPcoBinary();
  // Same genuine constant-color compiler output used by the native GS API
  // fixtures: vec4(.25, .5, .75, 1), independent of interpolated varyings.
  std::vector<std::uint8_t> fs = {
      0x35,0x8a,0x00,0x87,0x8c,0x01,0x00,0x00,0x00,0x20,0x35,0x8a,
      0x00,0x87,0x8b,0x01,0x00,0x00,0x00,0x21,0x86,0x92,0x40,0x13,
      0x00,0x00,0x40,0x3f,0x00,0x00,0x40,0xff,0x34,0x8a,0x00,0x87,
      0x40,0x00,0x00,0x22,0x38,0x8a,0x80,0x87,0x80,0x01,0x00,0x00,
      0x00,0x23,0xf3,0xff,0xff,0xff,0xff,0xff};
  auto gs = pvrgpu::stub::GeometryCompilerFixture(7); // native Layer = PrimitiveID & 1
  std::array<float,24> vertices = {
      -.75f,-.75f,0,1, .75f,-.75f,0,1, -.75f,.75f,0,1,
      -.75f,-.75f,0,1, .75f,-.75f,0,1, -.75f,.75f,0,1};
  std::array<std::uint32_t,16> vsh = {
      0x3f800000,0,0,0, 0,0x3f800000,0,0,
      0,0,0x3f800000,0, 0,0,0,0x3f800000};
  std::array<std::uint32_t,4> gsh{};
  const unsigned images = std::max(1U,layers);
  std::vector<std::uint8_t> initial(images*16*16*4);
  std::vector<float> initial_depth(images*16*16);
  for(unsigned l=0;l<images;++l) for(unsigned p=0;p<256;++p) {
    initial[(l*256+p)*4] = 17;
    initial[(l*256+p)*4+1] = static_cast<std::uint8_t>(31+l*19);
    initial[(l*256+p)*4+2] = static_cast<std::uint8_t>(71+l*23);
    initial[(l*256+p)*4+3] = 255;
    initial_depth[l*256+p] = l==0 ? .25f : 1.0f;
  }
  const auto expected_initial = initial;
  const auto expected_depth = initial_depth;
  pvrgpu_systemc_driver_command draw{};
  draw.version = PVRGPU_SYSTEMC_API_VERSION;
  draw.command = "draw_pco_triangles"; draw.case_name = "bridge.geometry.layered";
  draw.format = "PIPE_FORMAT_R8G8B8A8_UNORM"; draw.frame = 1;
  draw.width = draw.height = draw.framebuffer_width = draw.framebuffer_height = 16;
  draw.framebuffer_layers = layers;
  draw.vertex_pco=vs.data(); draw.vertex_pco_size=vs.size();
  draw.fragment_pco=fs.data(); draw.fragment_pco_size=fs.size();
  draw.geometry_pco=gs.data(); draw.geometry_pco_size=gs.size();
  draw.vertex_shared=vsh.data(); draw.vertex_shared_count=vsh.size();
  draw.geometry_shared=gsh.data(); draw.geometry_shared_count=gsh.size();
  draw.vertex_pco_abi={10,4,4,0,16,0,16,0,0,0};
  draw.fragment_pco_abi={1,0,0,4,0,0,0,0,0,0};
  draw.geometry_pco_abi={6,2,10,0,4,4,0,0,4,0};
  draw.raw_vertex_data=reinterpret_cast<const std::uint8_t*>(vertices.data());
  draw.raw_vertex_data_size=sizeof(vertices); draw.vertex_stride=16; draw.vertex_count=6;
  draw.vertex_attribute_count=1; draw.vertex_attribute_components[0]=4;
  draw.instance_count=1; draw.primitive_mode=4;
  draw.geometry_input_primitive_vertices=3; draw.geometry_output_primitive=5;
  draw.geometry_max_vertices=3; draw.geometry_invocations=1;
  draw.geometry_input_stride_dwords=4; draw.geometry_vertices_per_instance=6;
  draw.geometry_primitive_id_output_start=8; draw.geometry_primitive_id_output_count=1;
  draw.geometry_layer_output_start=9; draw.geometry_layer_output_count=1;
  draw.position_output_count=draw.varying_output_start=4;
  draw.fragment_position_count=draw.fragment_varying_start=4;
  draw.fragment_output_mask[0]=15;
  draw.viewport_scale_bits[0]=draw.viewport_scale_bits[1]=0x41000000;
  draw.viewport_scale_bits[2]=0x3f000000;
  std::memcpy(draw.viewport_translate_bits,draw.viewport_scale_bits,sizeof(draw.viewport_scale_bits));
  draw.half_pixel_center=draw.depth_clip_near=draw.depth_clip_far=1;
  draw.sample_mask=UINT32_MAX; draw.color_mask=15;
  draw.blend_source_rgb_factor=draw.blend_source_alpha_factor=PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
  draw.color_attachment_source_command_index=draw.depth_attachment_source_command_index=PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
  draw.initial_color_attachment_bytes=initial.data(); draw.initial_color_attachment_bytes_size=initial.size();
  if(depth) {
    draw.depth_format=pvrgpu::stub::kDriverPcoDepthFormatZ32Float;
    draw.depth_enable=draw.depth_write=1; draw.depth_func=1;
    draw.depth_clear_bits=0x3f800000;
    draw.initial_depth_attachment_bytes=reinterpret_cast<const std::uint8_t*>(initial_depth.data());
    draw.initial_depth_attachment_bytes_size=initial_depth.size()*4;
  }
  pvrgpu_systemc_driver_command sequence{}; sequence.version=PVRGPU_SYSTEMC_API_VERSION;
  sequence.command="draw_pco_sequence"; sequence.case_name=draw.case_name;
  sequence.format=draw.format; sequence.frame=1;
  sequence.width=sequence.height=sequence.framebuffer_width=sequence.framebuffer_height=16;
  sequence.pco_sequence_commands=&draw; sequence.pco_sequence_command_count=1;
  const auto jsonl=(root/"model.jsonl").string(), out=(root/"out").string();
  pvrgpu_systemc_submit_info info{};
  info.version=PVRGPU_SYSTEMC_API_VERSION; info.command=&sequence;
  info.jsonl_path=jsonl.c_str(); info.outdir=out.c_str();
  info.memory_mode=mode;
  std::array<char,1024> error{};
  const auto reject=[&](const char *field) {
    Check(pvrgpu_systemc_submit_driver_command(&info,error.data(),error.size())==2,
          std::string("reject layered boundary ")+field+": "+error.data());
  };
  sequence.framebuffer_layers=3; reject("logical tail"); sequence.framebuffer_layers=0;
  draw.framebuffer_layers=257; reject("layer bound"); draw.framebuffer_layers=layers;
  --draw.initial_color_attachment_bytes_size; reject("truncated LOAD"); ++draw.initial_color_attachment_bytes_size;
  draw.framebuffer_width=draw.framebuffer_height=4096; draw.framebuffer_layers=256;
  reject("attachment address slot"); draw.framebuffer_width=draw.framebuffer_height=16; draw.framebuffer_layers=layers;
  Check(!std::filesystem::exists(root),"rejected envelopes enqueue no work");
  std::filesystem::create_directories(root/"out");
  Check(pvrgpu_systemc_submit_driver_command(&info,error.data(),error.size())==0,
        std::string("accept native layered pipeline: ")+error.data());
  std::fill(initial.begin(),initial.end(),0); std::fill(initial_depth.begin(),initial_depth.end(),0);
  std::fill(vs.begin(),vs.end(),0); std::fill(gs.begin(),gs.end(),0); std::fill(fs.begin(),fs.end(),0);
  vertices.fill(0); vsh.fill(0); gsh.fill(UINT32_MAX); draw={};
  std::vector<std::uint8_t> pixels(expected_initial.size()+32,0xa5);
  pvrgpu_systemc_readback_info read{};
  read.version=PVRGPU_SYSTEMC_API_VERSION; read.width=read.height=16;
  read.bytes_per_pixel=4; read.layer_count=images;
  read.pixels=pixels.data()+16; read.pixels_size=expected_initial.size();
  --read.pixels_size;
  Check(pvrgpu_systemc_flush_readback(&read,error.data(),error.size())==2 && !read.pixels_written,
        "short all-layer readback rejected before model execution"); ++read.pixels_size;
  if(images>1) {
    read.layer_count=1;
    Check(pvrgpu_systemc_flush_readback(&read,error.data(),error.size())==0 && !read.pixels_written,
          std::string("partial layer readback cannot alias whole attachment: ")+error.data());
    Check(std::all_of(pixels.begin(),pixels.end(),[](auto v){return v==0xa5;}),"mismatched layer request leaves destination unchanged");
    read.layer_count=images;
  }
  Check(pvrgpu_systemc_flush_readback(&read,error.data(),error.size())==0 && read.pixels_written,
        std::string("native layered readback: ")+error.data());
  std::array<unsigned,3> covered{};
  for(unsigned l=0;l<images;++l) for(unsigned p=0;p<256;++p) {
    const auto offset=(l*256+p)*4;
    const auto *rgba=pixels.data()+16+offset;
    const bool red=rgba[0]==64&&rgba[1]==128&&rgba[2]==191&&rgba[3]==255;
    const bool written=l<2&&(!depth||l==1);
    Check((written&&red)||std::equal(rgba,rgba+4,expected_initial.data()+offset),
          "only native GS selected layer pixels may change layer="+std::to_string(l)+
          " pixel="+std::to_string(p)+" rgba="+std::to_string(rgba[0])+","+
          std::to_string(rgba[1])+","+std::to_string(rgba[2])+","+std::to_string(rgba[3]));
    covered[l]+=red;
    if(p==4*16+4) Check(red==written,"each layer has independent coverage and depth ownership");
    if(l==1&&!depth) {
      const bool layer0red=pixels[16+p*4]==64;
      Check(red==layer0red,"same XY in different layers has identical coverage without HSR collision");
    }
  }
  for(unsigned l=0;l<images;++l)
    Check((l<2&&(!depth||l==1)) ? covered[l]>30 : covered[l]==0,"full layer color-write count");
  for(unsigned i=0;i<16;++i)
    Check(pixels[i]==0xa5&&pixels[pixels.size()-1-i]==0xa5,"readback preserves both guards");
  if(depth) {
    std::vector<float> depths(initial_depth.size(),-1);
    read.attachment=UINT32_MAX; read.depth_format=pvrgpu::stub::kDriverPcoDepthFormatZ32Float;
    read.pixels=reinterpret_cast<std::uint8_t*>(depths.data()); read.pixels_size=depths.size()*4;
    Check(pvrgpu_systemc_flush_readback(&read,error.data(),error.size())==0&&read.pixels_written,"layered depth readback");
    for(unsigned l=0;l<images;++l) for(unsigned p=0;p<256;++p)
      Check(depths[l*256+p]==((l==1&&pixels[16+(l*256+p)*4]==64)?.5f:expected_depth[l*256+p]),"depth write affects only passing layer/sample");
  }
  std::ifstream log(jsonl); const std::string text{std::istreambuf_iterator<char>(log),{}};
  for(const char *evidence:{"\"gs_invocations\":2","\"gs_emitted_vertices\":6","\"pool_leaks\":0","\"pool_bytes_in_flight\":0"})
    Check(text.find(evidence)!=std::string::npos,std::string("native execution evidence: ")+evidence);
  if(layers)
    Check(text.find("\"driver_command_framebuffer_layers\":"+std::to_string(layers))!=std::string::npos,
          "completed native record identifies the full layered attachment");
  std::filesystem::remove_all(root);
}
}
int main(int argc,char **argv) {
  try {
    static_assert(PVRGPU_SYSTEMC_API_VERSION==30);
    const char *mode=argc>1?argv[1]:"direct";
    const auto root=std::filesystem::temp_directory_path()/("pvrgpu-layered-api-"+
        std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    Run(root/"three",mode,3,false); Run(root/"depth",mode,3,true);
    Run(root/"one",mode,1,false); Run(root/"nonlayered",mode,0,false);
    std::filesystem::remove(root);
    std::cout<<"layered API30 "<<mode<<" "<<checks<<" checks PASS\n";
    return 0;
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
