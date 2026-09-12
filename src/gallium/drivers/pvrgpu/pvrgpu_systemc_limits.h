/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_SYSTEMC_LIMITS_H
#define PVRGPU_SYSTEMC_LIMITS_H

#include <stdint.h>

/* Shared transport limit, not an estimate of total process heap usage. */
#define PVRGPU_SYSTEMC_MAX_PCO_SEQUENCE_PAYLOAD_BYTES (UINT64_C(512) * 1024u * 1024u)
/* GLES 3.x requires at least sixteen texture image units in every graphics
 * shader stage.  A combined image/sampler descriptor occupies 20 DWORDs, so
 * the public graphics transport must carry the complete 16 * 20 window.
 * This is a software transport bound, not a claim about a physical Rogue
 * shared-register file. */
#define PVRGPU_SYSTEMC_MAX_PCO_TEXTURES_PER_STAGE 16u
/* Keep a 64-DWORD suffix for UBO descriptors or packed live uniforms used by
 * otherwise valid sixteen-texture shaders. */
#define PVRGPU_SYSTEMC_MAX_PCO_GRAPHICS_SHARED_DWORDS_PER_STAGE 384u

#endif
