// HRX (ROCm/hrx-system libhrx) launch-overhead benchmark, the counterpart of
// bench/roofline/launch.hip. See docs/forks/amd-hrx-bench.md for results.
//
// usage: launch_hrx <kernels.hsaco>
//
// Measures libhrx exactly as shipped: the AMDGPU HAL driver's default AQL
// command-buffer mode. The PM4 mode is an IREE flag (--amdgpu_command_buffer_mode)
// that libhrx never parses, so it is not selectable from this API and is not
// measured. Builds against both the v0.3.0 release header (HRX_OLD_LOAD_API:
// 5-arg hrx_executable_load_data, no graph API) and HEAD (target family/key,
// graph API under HRX_HAS_GRAPHS).
#include <hrx/hrx_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}
static const char* msg(hrx_status_t s) {
  static char* m;
  size_t l = 0;
  if (m) hrx_status_free_message(m);
  m = NULL;
  hrx_status_to_string(s, &m, &l);
  return m ? m : "(no message)";
}
static void ck(hrx_status_t s, const char* what, int line) {
  if (hrx_status_is_ok(s)) return;
  fprintf(stderr, "HRX error line %d: %s: %s\n", line, what, msg(s));
  exit(1);
}
#define CK(x) ck((x), #x, __LINE__)

static hrx_executable_t exe;
static uint32_t ord_empty, ord_small, ord_store;
static hrx_buffer_ref_t bind_d;
static hrx_dispatch_config_t cfg_empty = {{1, 1, 1}, {64, 1, 1}, 0};
static hrx_dispatch_config_t cfg_small = {{256, 1, 1}, {256, 1, 1}, 0};
static uint32_t small_n = 65536;

static void dispatch(hrx_stream_t s, int small) {
  if (small)
    CK(hrx_stream_dispatch(s, exe, ord_small, &cfg_small, &small_n, sizeof small_n, &bind_d, 1, HRX_DISPATCH_FLAG_NONE));
  else
    CK(hrx_stream_dispatch(s, exe, ord_empty, &cfg_empty, NULL, 0, &bind_d, 1, HRX_DISPATCH_FLAG_NONE));
}

// N dispatches on one stream. flush_each=1 submits after every dispatch (one
// HSA queue submission per kernel, like a HIP stream launch); flush_each=0
// records all N into libhrx's pending one-shot command buffer and submits
// once at the synchronize (libhrx's native behaviour).
static void bench_stream(hrx_stream_t s, int small, int N, int flush_each) {
  double best = 1e30, best_enq = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    double t0 = now_us();
    for (int i = 0; i < N; ++i) {
      dispatch(s, small);
      if (flush_each) CK(hrx_stream_flush(s));
    }
    double t1 = now_us();
    CK(hrx_stream_synchronize(s));
    double t2 = now_us();
    double per = (t2 - t0) / N, enq = (t1 - t0) / N;
    if (per < best) best = per;
    if (enq < best_enq) best_enq = enq;
  }
  printf("stream  %-5s %-9s %-16s N=%d : %.2f us/dispatch (host record %.2f us)\n", small ? "small" : "empty",
         small ? "256x256" : "1x64", flush_each ? "flush-per-dispatch" : "batched,1-flush", N, best, best_enq);
}

#ifdef HRX_HAS_GRAPHS
static void bench_graph(hrx_device_t dev, hrx_stream_t s, int small, int nodes, int reps) {
  hrx_graph_t g;
  CK(hrx_graph_create(dev, 0, &g));
  hrx_graph_node_t prev = NULL;
  for (int i = 0; i < nodes; ++i) {
    hrx_graph_kernel_node_attrs_t a = {0};
    a.executable = exe;
    a.export_ordinal = small ? ord_small : ord_empty;
    a.config = small ? cfg_small : cfg_empty;
    a.constants = small ? &small_n : NULL;
    a.constants_size = small ? sizeof small_n : 0;
    a.bindings = &bind_d;
    a.binding_count = 1;
    a.flags = HRX_DISPATCH_FLAG_NONE;
    hrx_graph_node_t n;
    CK(hrx_graph_add_kernel_node(g, prev ? &prev : NULL, prev ? 1 : 0, &a, &n));
    prev = n;
  }
  hrx_graph_exec_t ge;
  double ti0 = now_us();
  CK(hrx_graph_instantiate(g, 0, &ge));
  double ti1 = now_us();
  CK(hrx_graph_exec_launch(ge, s));
  CK(hrx_stream_synchronize(s));
  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    double t0 = now_us();
    for (int r = 0; r < reps; ++r) CK(hrx_graph_exec_launch(ge, s));
    CK(hrx_stream_synchronize(s));
    double t1 = now_us();
    double per = (t1 - t0) / reps / nodes;
    if (per < best) best = per;
  }
  printf("graph   %-5s %4d nodes x %3d replays        : %.2f us/node (%.3f ms per replay, instantiate %.1f ms)\n",
         small ? "small" : "empty", nodes, reps, best, best * nodes / 1000.0, (ti1 - ti0) / 1000.0);
  hrx_graph_exec_release(ge);
  hrx_graph_release(g);
}
#endif

int main(int argc, char** argv) {
  const char* hsaco = argc > 1 ? argv[1] : "build/hrx/kernels.hsaco";
  int maj, min, pat;
  hrx_runtime_version(&maj, &min, &pat);
  printf("libhrx %d.%d.%d (shipped AQL command-buffer mode)\n", maj, min, pat);

  CK(hrx_gpu_initialize(0));
  int n = 0;
  CK(hrx_gpu_device_count(&n));
  hrx_device_t dev;
  CK(hrx_gpu_device_get(0, &dev));
  char name[128] = {0}, arch[64] = {0};
  CK(hrx_device_get_property(dev, HRX_DEVICE_PROPERTY_NAME, name, sizeof name));
  CK(hrx_device_get_property(dev, HRX_DEVICE_PROPERTY_ARCHITECTURE, arch, sizeof arch));
  char* colon = strchr(arch, ':');
  if (colon) *colon = 0;
  printf("device 0/%d: %s (%s)\n", n, name, arch);

  FILE* f = fopen(hsaco, "rb");
  if (!f) { perror(hsaco); return 1; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  void* data = malloc(sz);
  if (fread(data, 1, sz, f) != (size_t)sz) return 1;
  fclose(f);
#ifdef HRX_OLD_LOAD_API
  {  // v0.3.0: (device, data, size, executable_format, out). Format name from
     // env HRX_EXE_FORMAT or tried from a candidate list.
    const char* cands[] = {getenv("HRX_EXE_FORMAT"), "amdgcn-amd-amdhsa--gfx1151", "amdgcn-amd-amdhsa--gfx11-generic", "amdgcn-amd-amdhsa--", "amdgcn-amd-amdhsa", "amdgpu-hsaco", "hsaco", "amdgpu", "elf"};
    hrx_status_t st = NULL;
    for (size_t i = 0; i < sizeof cands / sizeof *cands && !exe; ++i) {
      if (!cands[i]) continue;
      st = hrx_executable_load_data(dev, data, sz, cands[i], &exe);
      if (hrx_status_is_ok(st)) printf("executable_format \"%s\" accepted\n", cands[i]);
      else { printf("executable_format \"%s\" rejected: %s\n", cands[i], msg(st)); hrx_status_ignore(st); }
    }
    if (!exe) return 1;
  }
#else
  CK(hrx_executable_load_data(dev, data, sz, "amdgpu", arch, &exe));
#endif
  CK(hrx_executable_lookup_export_by_name(exe, "hrx_empty", &ord_empty));
  CK(hrx_executable_lookup_export_by_name(exe, "hrx_small", &ord_small));
  CK(hrx_executable_lookup_export_by_name(exe, "hrx_store", &ord_store));
  const char* names[3] = {"hrx_empty", "hrx_small", "hrx_store"};
  const uint32_t ords[3] = {ord_empty, ord_small, ord_store};
  for (uint32_t k = 0; k < 3; ++k) {
    uint32_t o = ords[k];
    hrx_executable_export_info_t ei;
    CK(hrx_executable_export_info(exe, o, &ei));
    ei.name = names[k];  // v0.3.0 returns a non-NUL-terminated name
#ifdef HRX_OLD_LOAD_API
    printf("export %u %-10s constants %u x u32, bindings %u, wg %ux%ux%u\n", o, ei.name, ei.constant_count,
           ei.binding_count, ei.workgroup_size[0], ei.workgroup_size[1], ei.workgroup_size[2]);
#else
    printf("export %u %-10s constants %u B, bindings %u, wg %ux%ux%u\n", o, ei.name, ei.constant_byte_length,
           ei.binding_count, ei.workgroup_size[0], ei.workgroup_size[1], ei.workgroup_size[2]);
#endif
  }

  hrx_stream_t s;
  CK(hrx_stream_create(dev, 0, &s));
  const size_t BYTES = 1u << 24;
  hrx_buffer_t d;
  CK(hrx_buffer_allocate(s, BYTES, HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_BUFFER_USAGE_DEFAULT, &d));
  void* zeros = calloc(1, BYTES);
  CK(hrx_synchronous_h2d(dev, zeros, d, 0, BYTES));
  bind_d = (hrx_buffer_ref_t){d, 0, BYTES};

  // Correctness probe 1: HAL ABI (pointer args as bindings, scalars as constants).
  uint32_t want = 0xFEED1234u, got = 0;
  hrx_buffer_ref_t b1 = {d, 0, 4};
  hrx_dispatch_config_t c1 = {{1, 1, 1}, {1, 1, 1}, 0};
  CK(hrx_stream_dispatch(s, exe, ord_store, &c1, &want, sizeof want, &b1, 1, HRX_DISPATCH_FLAG_NONE));
  CK(hrx_stream_synchronize(s));
  CK(hrx_synchronous_d2h(dev, d, 0, &got, 4));
  printf("store via HAL ABI            : %s (0x%08x)\n", got == want ? "OK" : "MISMATCH", got);
  // Correctness probe 2: CUSTOM_DIRECT_ARGUMENTS (raw HIP-style kernarg blob).
  void* dptr = NULL;
  hrx_status_t st = hrx_buffer_get_device_ptr(d, &dptr);
  if (!hrx_status_is_ok(st)) {
    printf("store via CUSTOM_DIRECT_ARGS : skipped, hrx_buffer_get_device_ptr: %s\n", msg(st));
    hrx_status_ignore(st);
    st = hrx_make_status(HRX_STATUS_UNAVAILABLE, "no device pointer");
  } else {
    struct { void* p; uint32_t v; uint32_t pad; } raw = {dptr, 0xC0FFEE42u, 0};
    st = hrx_stream_dispatch(s, exe, ord_store, &c1, &raw, 12, NULL, 0, HRX_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  }
  if (dptr && hrx_status_is_ok(st)) {
    CK(hrx_stream_synchronize(s));
    CK(hrx_synchronous_d2h(dev, d, 0, &got, 4));
    printf("store via CUSTOM_DIRECT_ARGS : %s (0x%08x)\n", got == 0xC0FFEE42u ? "OK" : "MISMATCH", got);
  } else if (dptr) {
    printf("store via CUSTOM_DIRECT_ARGS : rejected: %s\n", msg(st));
    hrx_status_ignore(st);
  } else {
    hrx_status_ignore(st);
  }
  CK(hrx_synchronous_h2d(dev, zeros, d, 0, BYTES));

  // Warm-up.
  for (int i = 0; i < 100; ++i) dispatch(s, 0);
  CK(hrx_stream_synchronize(s));

  const int N = 10000;
  bench_stream(s, 0, N, 0);
  bench_stream(s, 0, N, 1);
  bench_stream(s, 1, N, 0);
  bench_stream(s, 1, N, 1);
#ifdef HRX_HAS_GRAPHS
  bench_graph(dev, s, 0, 100, 100);
  bench_graph(dev, s, 0, 2000, 20);
  bench_graph(dev, s, 1, 100, 100);
  bench_graph(dev, s, 1, 2000, 20);
#else
  printf("graph API not present in this libhrx build (v0.3.0 release predates it)\n");
#endif

  // Sanity on the small kernel's data: p -> p*0.999+1 converges to 1000.
  float v[3] = {0, 0, 0};
  CK(hrx_synchronous_d2h(dev, d, 0, &v[0], 4));
  CK(hrx_synchronous_d2h(dev, d, 65535 * 4, &v[1], 4));
  CK(hrx_synchronous_d2h(dev, d, 65536 * 4, &v[2], 4));
  printf("small-kernel data: p[0]=%.1f p[65535]=%.1f p[65536]=%.1f (last must stay 0)\n", v[0], v[1], v[2]);

  hrx_buffer_release(d);
  hrx_stream_release(s);
  hrx_executable_release(exe);
  hrx_device_release(dev);
  CK(hrx_gpu_shutdown());
  return 0;
}
