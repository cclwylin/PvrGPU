// SPDX-License-Identifier: MIT
// A sampler-state proof, never an assumption about texture contents or the
// coordinates of a lane that did not execute this texture instruction.
#include "texture/texture_filter.h"
#include <array>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace pvrgpu::stub;
unsigned checks=0;
void Check(bool b,const char *what){++checks;if(!b)throw std::runtime_error(what);}
struct RestoreRounding {
  int mode=std::fegetround();
  ~RestoreRounding(){std::fesetround(mode);}
};
void Equivalent(const RogueTextureSamplerDescriptor &s,unsigned mips){
  const auto base=SelectTextureLevels(SelectTextureLod(0,s,mips),s,mips);
  Check(base.level0==0&&base.level1==0,"independent selector keeps original base mip");
  for(int exponent=-120;exponent<=120;exponent+=3){
    const float rho=std::ldexp(1.F,exponent);
    const auto actual=SelectTextureLevels(SelectTextureLod(rho,s,mips),s,mips);
    Check(actual.level0==base.level0&&actual.level1==base.level1&&
          actual.image_filter==base.image_filter&&actual.mip_weight==base.mip_weight&&
          actual.mip_weight_u8==base.mip_weight_u8&&TextureLevelTaps(actual)==TextureLevelTaps(base),
          "every tested footprint has identical actual image/level/tap selection");
  }
}
void Run(){
  RestoreRounding restore;Check(std::fesetround(FE_TONEAREST)==0,"set nearest rounding");
  RogueTextureImageDescriptor image;image.mip_count=12;
  RogueTextureSamplerDescriptor sampler;
  for(auto filter:{TextureFilter::kNearest,TextureFilter::kLinear})
    for(unsigned mips:{2U,3U,12U,15U})for(unsigned max=1;max<32;++max)
      for(unsigned min:{0U,max/2,max}){
        image.mip_count=mips;sampler.min_filter=sampler.mag_filter=filter;
        sampler.mip_filter=TextureFilter::kNearest;sampler.min_lod_u4_6=min;sampler.max_lod_u4_6=max;
        Check(!TextureImplicitLodAffectsSelection(image,sampler),"bounded equal-filter base mip does not depend on derivatives");
        Equivalent(sampler,mips);
        Check(sampler.max_lod_u4_6==max&&sampler.min_lod_u4_6==min,"sampler window is never rewritten");
      }
  image.mip_count=12;sampler.min_filter=sampler.mag_filter=TextureFilter::kLinear;
  sampler.mip_filter=TextureFilter::kNearest;sampler.min_lod_u4_6=0;
  for(auto filter:{TextureFilter::kNearest,TextureFilter::kLinear})for(unsigned max:{1U,16U,31U}){
    sampler.min_filter=sampler.mag_filter=filter;sampler.max_lod_u4_6=max;
    const auto base=SelectTextureLevels(SelectTextureLod(0,sampler,image.mip_count),sampler,image.mip_count);
    for(float bias:{-4.F,0.F,4.F,-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
      for(float rho:{0.F,std::numeric_limits<float>::min(),1.F,256.F}){
        const auto selected=SelectTextureLevels(SelectTextureBiasedLod(rho,bias,sampler,image.mip_count),sampler,image.mip_count);
        Check(selected.level0==0&&selected.level1==0&&selected.image_filter==base.image_filter&&TextureLevelTaps(selected)==TextureLevelTaps(base),"actual shader bias policy remains inside same proved base selection");
      }
  }
  sampler.min_filter=sampler.mag_filter=TextureFilter::kLinear;
  for(unsigned max:{32U,33U,64U,128U,4095U}){
    sampler.max_lod_u4_6=max;
    Check(TextureImplicitLodAffectsSelection(image,sampler),"half-level boundary and wider window stay conservative");
  }
  sampler.max_lod_u4_6=16;
  sampler.mip_filter=TextureFilter::kLinear;
  Check(TextureImplicitLodAffectsSelection(image,sampler),"linear-mip blend still requires derivatives");
  sampler.mip_filter=TextureFilter::kNearest;sampler.min_filter=TextureFilter::kNearest;
  Check(TextureImplicitLodAffectsSelection(image,sampler),"different min/mag still require derivatives");
  sampler.min_filter=TextureFilter::kLinear;sampler.mag_filter=TextureFilter::kNearest;
  Check(TextureImplicitLodAffectsSelection(image,sampler),"reverse different filters require derivatives");
  sampler.min_filter=sampler.mag_filter=TextureFilter::kLinear;
  sampler.min_lod_u4_6=64;sampler.max_lod_u4_6=80;
  Check(TextureImplicitLodAffectsSelection(image,sampler),"fixed higher mip is deliberately outside new subset");
  sampler.min_lod_u4_6=0;sampler.max_lod_u4_6=16;
  for(int mode:{FE_TONEAREST,FE_DOWNWARD,FE_TOWARDZERO,FE_UPWARD}){
    Check(std::fesetround(mode)==0,"set actual selector rounding mode");
    const bool needed=TextureImplicitLodAffectsSelection(image,sampler);
    Check(needed==(mode==FE_UPWARD),"endpoint proof honors actual nearbyint rounding instead of assuming nearest");
    Check(std::fegetround()==mode,"dependency predicate preserves rounding mode");
    if(!needed)Equivalent(sampler,image.mip_count);
  }
  Check(std::fesetround(FE_TONEAREST)==0,"restore nearest for controls");
  image.mip_count=1;sampler.max_lod_u4_6=4095;
  for(auto mip:{TextureFilter::kNearest,TextureFilter::kLinear}){
    sampler.mip_filter=mip;Check(!TextureImplicitLodAffectsSelection(image,sampler),"legacy single-level equal-filter path preserved");
  }
  image.mip_count=12;sampler.max_lod_u4_6=0;sampler.min_filter=TextureFilter::kNearest;
  Check(!TextureImplicitLodAffectsSelection(image,sampler),"legacy exact-zero clamp still pins magnification selection");
  const auto reject=[&]{bool failed=false;try{(void)TextureImplicitLodAffectsSelection(image,sampler);}catch(const std::runtime_error&){failed=true;}Check(failed,"invalid selection metadata refuses");};
  sampler.min_lod_u4_6=1;reject();sampler.min_lod_u4_6=0;
  image.mip_count=0;reject();image.mip_count=16;reject();
}
}
int main(){try{Run();std::cout<<"LOD selection invariance: "<<checks<<" checks PASS\n";return 0;}catch(const std::exception&e){std::cerr<<"LOD invariance after "<<checks<<" checks: "<<e.what()<<'\n';return 1;}}
