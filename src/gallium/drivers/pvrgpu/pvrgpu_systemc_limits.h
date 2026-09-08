/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_SYSTEMC_LIMITS_H
#define PVRGPU_SYSTEMC_LIMITS_H

#include <stdint.h>

/* Shared transport limit, not an estimate of total process heap usage. */
#define PVRGPU_SYSTEMC_MAX_PCO_SEQUENCE_PAYLOAD_BYTES (UINT64_C(512) * 1024u * 1024u)

#endif
