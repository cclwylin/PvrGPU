/* SPDX-License-Identifier: MIT */
/* The actual recorder's CB0 reader and private packed-word transport.
 * No model or compiler stubs and no shader execution in this unit test. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_context.c"

static unsigned checks;
#define CHECK(c) do { ++checks; if (!(c)) { \
   fprintf(stderr,"%s:%u: %s\n",__FILE__,__LINE__,#c); exit(1); \
} } while (0)

static uint32_t payload(unsigned i)
{
   static const uint32_t edge[] = {0,0x80000000,0x7fc12345,0xff800000,
      0x7f800000,0xffffffff,0x00000001,0x7fffffff};
   return i < ARRAY_SIZE(edge) ? edge[i] : i * UINT32_C(0x9e3779b9) + 37;
}
static void fill(uint32_t *words, size_t count)
{ for (size_t i = 0; i < count; ++i) words[i] = UINT32_C(0xa5000000) + i; }

static void test_maps_and_lifetime(void)
{
   uint8_t raw[13+512*4+17], original[sizeof(raw)];
   memset(raw,0xe3,sizeof(raw));
   for (unsigned i = 0; i < 512; ++i) {
      uint32_t value = payload(i); memcpy(raw+13+i*4,&value,4);
   }
   memcpy(original,raw,sizeof(raw));
   struct pvrgpu_context ctx = {0};
   struct pvrgpu_resource resource = {0};
   resource.base.target = PIPE_BUFFER; resource.data = raw; resource.size = sizeof(raw)-17;
   const mesa_shader_stage stages[] = {MESA_SHADER_VERTEX,MESA_SHADER_FRAGMENT,
      MESA_SHADER_GEOMETRY,MESA_SHADER_TESS_CTRL,MESA_SHADER_TESS_EVAL};
   for (unsigned extent = 4; extent <= 512; extent += 4) {
      struct pvrgpu_pco_uniform_word_map map = {.source_dwords=extent};
      for (unsigned i = 0; i < extent; ++i)
         if (i%7 == 0 || (i%4 == 1 && extent <= 256)) map.source_words[map.count++] = i;
      CHECK(map.count <= 256);
      struct pvrgpu_pco_stage_abi abi = {.shareds=256,
         .push_constant_start=256-map.count,.push_constant_count=map.count};
      for (unsigned backing = 0; backing < 2; ++backing) {
         for (unsigned stage = 0; stage < ARRAY_SIZE(stages); ++stage) {
            struct pipe_constant_buffer *binding = &ctx.constant_buffers[stages[stage]][0];
            *binding = (struct pipe_constant_buffer){.buffer_size=extent*4,
               .buffer_offset=backing ? 13 : 29};
            if (backing) binding->buffer=&resource.base;
            else binding->user_buffer=raw+13; /* offset must not apply twice */
            uint32_t output[258], before[258]; fill(output,ARRAY_SIZE(output)); memcpy(before,output,sizeof(output));
            CHECK(pvrgpu_stage_uniform_dwords(&ctx,stages[stage]) == extent);
            CHECK(pvrgpu_copy_stage_uniform_words(&ctx,stages[stage],&abi,&map,output+1));
            CHECK(output[0] == before[0] && output[257] == before[257]);
            for (unsigned i = 0; i < abi.push_constant_start; ++i) CHECK(output[1+i] == before[1+i]);
            for (unsigned i = 0; i < map.count; ++i) CHECK(output[1+abi.push_constant_start+i] == payload(map.source_words[i]));
            CHECK(memcmp(raw,original,sizeof(raw)) == 0);
         }
      }
   }
   /* Actual Draw360 source shape: retain all four sparse scalar array words. */
   struct pvrgpu_pco_uniform_word_map map = {.source_dwords=108};
   for (unsigned i = 0; i < 108; ++i)
      if (i < 12 || (i >= 16 && i <= 86) || (i >= 88 && i <= 91) ||
          i == 92 || i == 96 || i == 100 || i == 104) map.source_words[map.count++] = i;
   CHECK(map.count == 91 && map.source_words[90] == 104);
   struct pvrgpu_pco_stage_abi abi = {.shareds=255,.push_constant_start=164,.push_constant_count=91};
   ctx.constant_buffers[MESA_SHADER_FRAGMENT][0]=(struct pipe_constant_buffer){.user_buffer=raw+13,.buffer_size=432};
   uint32_t output[256], before[256]; fill(output,256); memcpy(before,output,sizeof(output));
   CHECK(pvrgpu_copy_stage_uniform_words(&ctx,MESA_SHADER_FRAGMENT,&abi,&map,output));
   for (unsigned i = 0; i < 164; ++i) CHECK(output[i] == before[i]);
   for (unsigned i = 0; i < 91; ++i) CHECK(output[164+i] == payload(map.source_words[i]));
   CHECK(output[255] == before[255]);
   memset(raw,0x6a,sizeof(raw)); memset(&ctx,0,sizeof(ctx));
   for (unsigned i = 0; i < 91; ++i) CHECK(output[164+i] == payload(map.source_words[i]));
}

static void test_padding_and_legacy(void)
{
   uint8_t raw[31]; for (unsigned i=0;i<sizeof(raw);++i) raw[i]=(uint8_t)(i*13+7);
   struct pvrgpu_context ctx={0};
   struct pvrgpu_pco_uniform_word_map zero={0};
   for (unsigned bytes=1;bytes<=sizeof(raw);++bytes) {
      ctx.constant_buffers[MESA_SHADER_FRAGMENT][0]=(struct pipe_constant_buffer){.user_buffer=raw,.buffer_size=bytes};
      const unsigned extent=((bytes/4)+3)&~3U;
      struct pvrgpu_pco_stage_abi abi={.shareds=16,.push_constant_start=8,.push_constant_count=8};
      uint32_t legacy[16],mapped[16]; fill(legacy,16); fill(mapped,16);
      CHECK(pvrgpu_copy_stage_uniform_words(&ctx,MESA_SHADER_FRAGMENT,&abi,NULL,legacy));
      CHECK(pvrgpu_copy_stage_uniform_words(&ctx,MESA_SHADER_FRAGMENT,&abi,&zero,mapped));
      CHECK(memcmp(legacy,mapped,sizeof(legacy))==0);
      for(unsigned i=0;i<8;++i) {uint32_t want=0;if(i*4+4<=bytes)memcpy(&want,raw+i*4,4);CHECK(legacy[8+i]==want);}
      if(extent) {
         struct pvrgpu_pco_uniform_word_map map={.count=2,.source_dwords=extent,.source_words={0,extent-1}};
         abi.push_constant_count=2;fill(mapped,16);
         CHECK(pvrgpu_copy_stage_uniform_words(&ctx,MESA_SHADER_FRAGMENT,&abi,&map,mapped));
         CHECK(mapped[8]==legacy[8] && mapped[9]==legacy[8+extent-1]);
      }
   }
   /* Resource-backed truncation uses its real available span, not binding size. */
   struct pvrgpu_resource resource={0};resource.data=raw;resource.size=sizeof(raw);resource.base.target=PIPE_BUFFER;
   ctx.constant_buffers[MESA_SHADER_FRAGMENT][0]=(struct pipe_constant_buffer){.buffer=&resource.base,.buffer_offset=18,.buffer_size=100};
   struct pvrgpu_pco_uniform_word_map map={.count=2,.source_dwords=4,.source_words={0,3}};
   struct pvrgpu_pco_stage_abi abi={.shareds=6,.push_constant_start=4,.push_constant_count=2};
   uint32_t words[8];fill(words,8);uint32_t first;memcpy(&first,raw+18,4);
   CHECK(pvrgpu_stage_uniform_dwords(&ctx,MESA_SHADER_FRAGMENT)==4);
   CHECK(pvrgpu_copy_stage_uniform_words(&ctx,MESA_SHADER_FRAGMENT,&abi,&map,words));
   CHECK(words[4]==first && words[5]==0);
}

static void test_rejections_are_atomic(void)
{
   uint32_t source[108],output[256],before[256];fill(source,108);fill(before,256);
   struct pvrgpu_pco_uniform_word_map good={.count=4,.source_dwords=108,.source_words={0,4,5,104}};
   struct pvrgpu_pco_stage_abi original={.shareds=256,.push_constant_start=252,.push_constant_count=4};
   for(unsigned failure=0;failure<16;++failure) {
      struct pvrgpu_pco_uniform_word_map map=good;struct pvrgpu_pco_stage_abi abi=original;
      const uint8_t *bytes=(const uint8_t*)source;size_t available=sizeof(source),capacity=256;
      switch(failure) {
      case 0:map.count=257;break;
      case 1:map.count=3;break;
      case 2:map.source_dwords=0;break;
      case 3:map.source_dwords=104;break;
      case 4:map.source_words[3]=108;break;
      case 5:map.source_words[1]=0;break;
      case 6:map.source_words[2]=3;break;
      case 7:map.source_words[255]=1;break;
      case 8:map.count=0;break;
      case 9:memset(&map,0,sizeof(map));map.source_words[0]=1;break;
      case 10:abi.push_constant_start=UINT32_MAX;break;
      case 11:abi.shareds=255;break;
      case 12:capacity=255;break;
      case 13:bytes=NULL;break;
      case 14:available=SIZE_MAX;break;
      case 15:available=sizeof(source)-16;break;
      }
      memcpy(output,before,sizeof(output));
      CHECK(!pvrgpu_copy_push_constant_words(&abi,&map,bytes,available,output,capacity));
      CHECK(memcmp(output,before,sizeof(output))==0);
   }
   struct pvrgpu_pco_stage_abi empty={0};struct pvrgpu_pco_uniform_word_map zero={0};
   CHECK(pvrgpu_copy_push_constant_words(&empty,&zero,NULL,0,output,256));
   CHECK(!pvrgpu_copy_push_constant_words(&empty,&good,NULL,0,output,256));
   CHECK(!pvrgpu_copy_push_constant_words(NULL,&zero,NULL,0,output,256));
   CHECK(!pvrgpu_copy_push_constant_words(&empty,&zero,NULL,0,NULL,256));
}

static void test_full_destination_map(void)
{
   uint32_t source[512],original[512];
   for(unsigned i=0;i<512;++i)source[i]=payload(i);
   memcpy(original,source,sizeof(source));
   for(unsigned high=0;high<2;++high) {
      struct pvrgpu_context ctx={0};
      struct pvrgpu_pco_uniform_word_map map={.count=256,.source_dwords=high?512:256};
      for(unsigned i=0;i<256;++i)map.source_words[i]=i;
      if(high)map.source_words[255]=511;
      struct pvrgpu_pco_stage_abi abi={.shareds=256,.push_constant_count=256};
      ctx.constant_buffers[MESA_SHADER_FRAGMENT][0]=(struct pipe_constant_buffer){
         .user_buffer=source,.buffer_size=map.source_dwords*4};
      uint32_t output[258],before[258];fill(output,258);memcpy(before,output,sizeof(output));
      CHECK(pvrgpu_copy_stage_uniform_words(&ctx,MESA_SHADER_FRAGMENT,&abi,&map,output+1));
      CHECK(output[0]==before[0]&&output[257]==before[257]);
      for(unsigned i=0;i<256;++i)CHECK(output[1+i]==original[map.source_words[i]]);
      CHECK(memcmp(source,original,sizeof(source))==0);
   }
}

int main(void)
{
   test_maps_and_lifetime();test_padding_and_legacy();test_rejections_are_atomic();test_full_destination_map();
   printf("Push-constant word map: %u checks PASS\n",checks);return 0;
}
