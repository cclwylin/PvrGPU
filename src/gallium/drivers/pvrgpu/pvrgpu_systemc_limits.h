/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_SYSTEMC_LIMITS_H
#define PVRGPU_SYSTEMC_LIMITS_H

#include <stdint.h>

/* Shared transport limit, not an estimate of total process heap usage. */
#define PVRGPU_SYSTEMC_MAX_PCO_SEQUENCE_PAYLOAD_BYTES (UINT64_C(512) * 1024u * 1024u)
/* Twelve 20-DWORD combined descriptors fit the existing 256-DWORD largest
 * stage transport. Every stage still validates its own system/UBO/CB0 prefix
 * and shared-register budget; this is not a larger hardware register file. */
#define PVRGPU_SYSTEMC_MAX_PCO_TEXTURES_PER_STAGE 12u

#endif
