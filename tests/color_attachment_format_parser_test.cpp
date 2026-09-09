#include "driver_command.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
unsigned checks=0;
void Check(bool ok,const std::string &why){++checks;if(!ok)throw std::runtime_error(why);}
// Existing bounded conditionals command metadata. This is a parser test, not
// executable shader/pixel evidence; no PCO payload is synthesized here.
const std::string base=
  "schema=pvrgpu.driver-command.v1\nproducer=pvrgpu-gallium-driver\n"
  "command=draw_pco_triangles\ncase=glmark2.conditionals\nframe=1\n"
  "framebuffer_width=80\nframebuffer_height=60\nwidth=80\nheight=60\n"
  "format=PIPE_FORMAT_R8G8B8A8_UNORM\nclear_color_bits=0,0,0,1065353216\n"
  "raw_vertex_data_size=73728\nvertex_stride=12\nvertex_count=6144\nfirst_vertex=0\n"
  "instance_count=1\nprimitive_mode=4\nindexed=0\nrender_target_count=4\n"
  "vertex_pco_size=520\nfragment_pco_size=520\nvertex_shared_count=16\n"
  "vertex_shared_words=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
  "fragment_shared_count=4\nfragment_shared_words=0,0,0,0\n"
  "vertex_pco_abi=10,4,4,0,16,0,16,0\nfragment_pco_abi=4,0,0,0,4,0,4,0\n"
  "position_linkage=0,4,0,0\nviewport_scale_bits=1109393408,1106247680,1056964608\n"
  "viewport_translate_bits=1109393408,1106247680,1056964608\n"
  "raster_state=0,2,0,0,0,0,0,1,0,0,1,1,0\nscissor_rect=0,0,0,0\n"
  "primitive_width=1065353216,1065353216\npoint_size_output=0,0\n"
  "sample_mask=4294967295\ncolor_state=15,0,1\ndepth_state=1,1,3,1065353216,1\n";
const std::string rgba="PIPE_FORMAT_R8G8B8A8_UNORM";
const std::string rgb10="PIPE_FORMAT_R10G10B10A2_UNORM";
const std::string bgr10="PIPE_FORMAT_B10G10R10A2_UNORM";
}
int main(){
  try {
    const auto root=std::filesystem::temp_directory_path()/("pvrgpu-color-format-parser-"+std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    unsigned ordinal=0;
    const auto parse=[&](const std::string &tail,bool expected){
      const auto path=root/(std::to_string(ordinal++)+".txt");
      {std::ofstream out(path);out<<base<<tail;Check(out.good(),"write private metadata fixture");}
      pvrgpu::stub::DriverCommand command;std::string error;
      const bool accepted=pvrgpu::stub::LoadDriverCommand(path.string(),&command,&error);
      Check(accepted==expected,"parse case "+std::to_string(ordinal-1)+": "+error);
      if(accepted)Check(command.render_target_count==4,"preserve render target count");
      return command;
    };
    auto legacy=parse("",true);Check(legacy.color_attachment_formats.empty(),"legacy omitted fields preserved");
    const auto list=rgba+","+rgb10+","+bgr10+","+rgba;
    auto mixed=parse("color_attachment_format_count=4\ncolor_attachment_formats="+list+"\n",true);
    Check(mixed.color_attachment_formats==std::vector<std::string>{rgba,rgb10,bgr10,rgba},"exact owned ordered vector");
    for(const std::string &count:{"0","1","3","5","-1","4294967296","four"})
      parse("color_attachment_format_count="+count+"\ncolor_attachment_formats="+list+"\n",false);
    parse("color_attachment_format_count=4\n",false);
    parse("color_attachment_formats="+list+"\n",false);
    for(const std::string &bad:{std::string{},list+",",","+list,rgba+",,"+bgr10+","+rgba,
         rgb10+","+rgb10+","+bgr10+","+rgba,rgba+","+rgb10+","+bgr10,
         list+","+rgba,rgba+",PIPE_FORMAT_R32_UINT,"+bgr10+","+rgba,
         rgba+",PIPE_FORMAT_R8G8B8A8_SRGB,"+bgr10+","+rgba,
         rgba+", "+rgb10+","+bgr10+","+rgba})
      parse("color_attachment_format_count=4\ncolor_attachment_formats="+bad+"\n",false);
    parse("color_attachment_format_count=4\ncolor_attachment_formats="+list+"\ncolor_attachment_format_count=4\n",false);
    std::cout<<"color attachment parser: PASS "<<checks<<" checks; artifacts="<<root<<'\n';
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
