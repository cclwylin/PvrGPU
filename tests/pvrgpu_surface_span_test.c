/* SPDX-License-Identifier: MIT */
/* Bounds-only unit using the pinned Mesa pipe_resource/pipe_surface types. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_resource.h"
#include <limits.h>
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;
#define CHECK(c) do { ++checks; if (!(c)) { ++failures; \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c); } } while (0)

static void init(struct pvrgpu_resource *r, struct pipe_surface *s)
{
   memset(r,0,sizeof(*r)); memset(s,0,sizeof(*s));
   r->base.target=PIPE_TEXTURE_2D_ARRAY;
   r->base.width0=64;r->base.height0=32;r->base.array_size=8;
   r->base.last_level=2;r->level_count=3;
   r->level_offsets[0]=0;r->level_offsets[1]=32768;r->level_offsets[2]=65536;
   r->level_strides[1]=80;r->level_layer_strides[1]=512;
   r->size=65536;
   s->texture=&r->base;s->level=1;s->first_layer=2;s->last_layer=5;
}

static void check_span(const struct pvrgpu_resource *r,const struct pipe_surface *s,
                       size_t row,unsigned height,unsigned layers,
                       bool expected,size_t expected_offset)
{
   const size_t sentinel=SIZE_MAX-123;
   size_t offset=sentinel;
   bool result=pvrgpu_surface_span(r,s,row,height,layers,&offset);
   CHECK(result==expected);
   CHECK(offset==(expected?expected_offset:sentinel));
}

static void named_cases(void)
{
   struct pvrgpu_resource r;struct pipe_surface s;init(&r,&s);
   const size_t start=32768+2*512;
   check_span(&r,&s,64,6,4,true,start); // mip prefix + nonzero layer + row padding
   check_span(&r,&s,64,6,2,true,start); // a bounded subset of the surface view
   r.size=32768+5*512+5*80+64;
   check_span(&r,&s,64,6,4,true,start); // exact last byte, no unused tail required
   --r.size;check_span(&r,&s,64,6,4,false,0);
   init(&r,&s);r.level_layer_strides[1]=463;
   check_span(&r,&s,64,6,4,false,0); // adjacent layer would overlap live final row
   r.level_layer_strides[1]=464;r.size=32768+5*464+464;
   check_span(&r,&s,64,6,4,true,32768+2*464);
   init(&r,&s);r.level_strides[1]=63;check_span(&r,&s,64,6,4,false,0);
   r.level_strides[1]=0;check_span(&r,&s,64,6,4,false,0); // never divide by zero
   init(&r,&s);r.level_layer_strides[1]=0;check_span(&r,&s,64,6,4,false,0);
   init(&r,&s);r.level_offsets[1]=r.size+1;check_span(&r,&s,64,6,4,false,0);
   init(&r,&s);s.first_layer=6;check_span(&r,&s,64,6,1,false,0);
   init(&r,&s);check_span(&r,&s,64,6,5,false,0);
   init(&r,&s);s.level=3;check_span(&r,&s,64,6,4,false,0);
   s.level=PIPE_MAX_TEXTURE_LEVELS;r.level_count=UINT_MAX;
   check_span(&r,&s,64,6,4,false,0); // array index bounded independently of count
   init(&r,&s);check_span(&r,&s,0,6,4,false,0);
   check_span(&r,&s,64,0,4,false,0);check_span(&r,&s,64,6,0,false,0);
   s.first_layer=0;s.last_layer=256;
   check_span(&r,&s,64,6,257,false,0);
   check_span(NULL,&s,64,6,4,false,0);check_span(&r,NULL,64,6,4,false,0);
   CHECK(!pvrgpu_surface_span(&r,&s,64,6,4,NULL));
   init(&r,&s);r.size=SIZE_MAX;r.level_offsets[1]=SIZE_MAX-127;
   r.level_layer_strides[1]=128;s.first_layer=s.last_layer=1;
   check_span(&r,&s,1,1,1,false,0); // base + layer must not wrap
   init(&r,&s);r.size=SIZE_MAX;r.level_offsets[1]=0;
   r.level_layer_strides[1]=SIZE_MAX;s.first_layer=s.last_layer=0;
   check_span(&r,&s,64,1,1,true,0);
   s.first_layer=s.last_layer=2;check_span(&r,&s,64,1,1,false,0); // layer product overflow
   init(&r,&s);r.size=SIZE_MAX;r.level_offsets[1]=0;
   r.level_strides[1]=UINT_MAX;r.level_layer_strides[1]=SIZE_MAX;
   s.first_layer=s.last_layer=0;
   check_span(&r,&s,SIZE_MAX,UINT_MAX,1,false,0);
   check_span(&r,&s,UINT_MAX,UINT_MAX,1,true,0); // large but mathematically valid
}

/* Independent wide-arithmetic oracle: no division and no helper subroutines. */
static void generated_cases(void)
{
   uint32_t random=0x243f6a88;
   for(unsigned test=0;test<12000;++test) {
      struct pvrgpu_resource r;struct pipe_surface s;init(&r,&s);
      random=random*1664525u+1013904223u;
      size_t row=(random>>4)%180;unsigned height=(random>>12)%12;
      unsigned layers=(random>>20)%12;
      random=random*1664525u+1013904223u;
      r.level_offsets[1]=(random>>2)%240;
      r.level_strides[1]=(random>>10)%200;
      r.level_layer_strides[1]=(random>>19)%1300;
      random=random*1664525u+1013904223u;
      r.size=random%15000;s.first_layer=(random>>14)%8;s.last_layer=(random>>20)%12;
      const __uint128_t base=r.level_offsets[1],stride=r.level_strides[1];
      const __uint128_t layer_stride=r.level_layer_strides[1];
      bool expected=row && height && layers && s.first_layer<=s.last_layer &&
         layers<=(unsigned)s.last_layer-s.first_layer+1 && stride>=row && layer_stride;
      __uint128_t offset=base+(__uint128_t)s.first_layer*layer_stride;
      if(expected) {
         const __uint128_t used=(__uint128_t)(height-1)*stride+row;
         const __uint128_t end=offset+(__uint128_t)(layers-1)*layer_stride+used;
         expected=used<=layer_stride && end<=r.size;
      }
      check_span(&r,&s,row,height,layers,expected,(size_t)offset);
   }
}
int main(void)
{
   named_cases();generated_cases();
   printf("surface span: %s (%u checks, %u failures)\n",failures?"FAIL":"PASS",checks,failures);
   return failures?1:0;
}
