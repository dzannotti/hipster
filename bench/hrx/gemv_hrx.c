// HRX side of the gemv comparison: dispatches the engine's k_gemv_q8<Q4_K,4,1>
// (HIP-compiled HSACO, unbundled from `clang++ --offload-device-only`) through
// libhrx's HAL ABI: the 5 pointer args become bindings, {N,K} the constants,
// HIP hidden args (block count for gridDim.x) are filled by the runtime.
// usage: gemv_hrx <gemv_dev.hsaco> [N] [K]
#include <hrx/hrx_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3; }
static const char* msg(hrx_status_t s) { static char* m; size_t l = 0; if (m) hrx_status_free_message(m); m = NULL; hrx_status_to_string(s, &m, &l); return m ? m : "(no message)"; }
static void ck(hrx_status_t s, const char* what, int line) { if (hrx_status_is_ok(s)) return; fprintf(stderr, "HRX error line %d: %s: %s\n", line, what, msg(s)); exit(1); }
#define CK(x) ck((x), #x, __LINE__)
static uint32_t g_lcg = 12345u;
static uint32_t lcg(void) { g_lcg = g_lcg * 1664525u + 1013904223u; return g_lcg; }
static const char* KNAME = "_ZN3hip12_GLOBAL__N_19k_gemv_q8ILNS_4WFmtE0ELi4ELi1EEEvPKhPKaPKfS8_Pfii";

int main(int argc, char** argv) {
  const char* hsaco = argc > 1 ? argv[1] : "build/hrx/gemv_dev.hsaco";
  const int N = argc > 2 ? atoi(argv[2]) : 11648, K = argc > 3 ? atoi(argv[3]) : 5120, iters = 50;
  const size_t rb = (size_t)K / 256 * 144, wbytes = rb * N, nsb = K / 32;  // Q4_K: 144 B per 256 weights
  printf("HRX  k_gemv_q8<Q4_K,4,1> N=%d K=%d row_bytes=%zu W=%.1f MB grid=%d\n", N, K, rb, wbytes / 1048576.0, (N + 63) / 64);
  CK(hrx_gpu_initialize(0));
  hrx_device_t dev; CK(hrx_gpu_device_get(0, &dev));
  char arch[64] = {0}; CK(hrx_device_get_property(dev, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch, sizeof arch));
  char* colon = strchr(arch, ':'); if (colon) *colon = 0;
  FILE* f = fopen(hsaco, "rb"); if (!f) { perror(hsaco); return 1; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  void* data = malloc(sz); if (fread(data, 1, sz, f) != (size_t)sz) return 1; fclose(f);
  hrx_executable_t exe = NULL;
#ifdef HRX_OLD_LOAD_API
  { char fmt[96]; snprintf(fmt, sizeof fmt, "amdgcn-amd-amdhsa--%s", arch); CK(hrx_executable_load_data(dev, data, sz, fmt, &exe)); }
#else
  CK(hrx_executable_load_data(dev, data, sz, "amdgpu", arch, &exe));
#endif
  size_t nexp = 0; CK(hrx_executable_export_count(exe, &nexp));
  uint32_t ord = 0; CK(hrx_executable_lookup_export_by_name(exe, KNAME, &ord));
  hrx_executable_export_info_t ei; CK(hrx_executable_export_info(exe, ord, &ei));
#ifdef HRX_OLD_LOAD_API
  printf("loaded %zu exports; ordinal %u: constants %u x u32, bindings %u\n", nexp, ord, ei.constant_count, ei.binding_count);
#else
  printf("loaded %zu exports; ordinal %u: constants %u B, bindings %u\n", nexp, ord, ei.constant_byte_length, ei.binding_count);
#endif

  hrx_stream_t s; CK(hrx_stream_create(dev, 0, &s));
  hrx_buffer_t W, xq, xd, xs, y;
  CK(hrx_buffer_allocate(s, wbytes, HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &W));
  CK(hrx_buffer_allocate(s, K, HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &xq));
  CK(hrx_buffer_allocate(s, nsb * 4, HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &xd));
  CK(hrx_buffer_allocate(s, nsb * 4, HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &xs));
  CK(hrx_buffer_allocate(s, (size_t)N * 4, HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &y));
  uint8_t* hw = malloc(wbytes); for (size_t i = 0; i < wbytes; ++i) hw[i] = lcg() >> 24;
  int8_t* hq = malloc(K); for (int i = 0; i < K; ++i) hq[i] = (int8_t)(lcg() >> 24);
  float *hd = malloc(nsb * 4), *hs = malloc(nsb * 4);
  for (size_t i = 0; i < nsb; ++i) { hd[i] = ((lcg() >> 8) & 0xFFFF) / 65536.f * 0.01f; hs[i] = ((lcg() >> 8) & 0xFFFF) / 65536.f * 0.01f; }
  float* hy = calloc(N, 4);
  CK(hrx_synchronous_h2d(dev, hw, W, 0, wbytes)); CK(hrx_synchronous_h2d(dev, hq, xq, 0, K));
  CK(hrx_synchronous_h2d(dev, hd, xd, 0, nsb * 4)); CK(hrx_synchronous_h2d(dev, hs, xs, 0, nsb * 4));
  CK(hrx_synchronous_h2d(dev, hy, y, 0, (size_t)N * 4));

  hrx_buffer_ref_t b[5] = {{W, 0, wbytes}, {xq, 0, (size_t)K}, {xd, 0, nsb * 4}, {xs, 0, nsb * 4}, {y, 0, (size_t)N * 4}};
  int32_t consts[2] = {N, K};
  hrx_dispatch_config_t cfg = {{(uint32_t)((N + 63) / 64), 1, 1}, {256, 1, 1}, 0};
#define LAUNCH() CK(hrx_stream_dispatch(s, exe, ord, &cfg, consts, sizeof consts, b, 5, HRX_DISPATCH_FLAG_NONE))
  for (int i = 0; i < 5; ++i) LAUNCH();
  CK(hrx_stream_synchronize(s));
  for (int flush_each = 0; flush_each < 2; ++flush_each)
    for (int rep = 0; rep < 3; ++rep) {
      double t0 = now_us();
      for (int i = 0; i < iters; ++i) { LAUNCH(); if (flush_each) CK(hrx_stream_flush(s)); }
      CK(hrx_stream_synchronize(s));
      double us = now_us() - t0;
      printf("HRX  %s rep %d: %.1f us/dispatch, %.1f GB/s\n", flush_each ? "flush-per-dispatch" : "batched          ", rep, us / iters, wbytes * (double)iters / us / 1e3);
    }
  CK(hrx_synchronous_d2h(dev, y, 0, hy, (size_t)N * 4));
  double sum = 0; for (int i = 0; i < N; ++i) sum += hy[i];
  printf("HRX  checksum: sum=%.6e y[0]=%.6e y[N-1]=%.6e\n", sum, hy[0], hy[N - 1]);
  return 0;
}
