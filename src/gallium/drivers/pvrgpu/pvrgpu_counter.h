/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_COUNTER_H
#define PVRGPU_COUNTER_H

#ifdef __cplusplus
extern "C" {
#endif

#define PVRGPU_DRIVER_COUNTER_SCHEMA "pvrgpu.driver-counter.v1"

void
pvrgpu_counter_event(const char *event, const char *details);

void
pvrgpu_counter_eventf(const char *event, const char *format, ...);

#ifdef __cplusplus
}
#endif

#endif /* PVRGPU_COUNTER_H */
