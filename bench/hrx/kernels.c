// Kernels for bench/hrx/launch_hrx.c, compiled straight to a gfx1151 HSACO the
// way libhrx's CTS does (libhrx/cts/tests/executable/executable_kernels.c):
// plain C with [[clang::amdgpu_kernel]], no HIP hidden-argument suffix.
#define KERNEL [[clang::amdgpu_kernel, gnu::visibility("protected")]]

// Same shape as bench/roofline/launch.hip k_empty: one wave, touches nothing.
KERNEL void hrx_empty(float* p) {
  if (__builtin_amdgcn_workitem_id_x() == 0 && __builtin_amdgcn_workgroup_id_x() == 0 && p[0] == 12345.f) p[1] = 1.f;
}

// Same as launch.hip k_small: grid 256x256, p[i] = p[i]*0.999 + 1 (n = 65536).
KERNEL void hrx_small(float* p, unsigned n) {
  unsigned i = __builtin_amdgcn_workgroup_id_x() * 256u + __builtin_amdgcn_workitem_id_x();
  if (i < n) p[i] = p[i] * 0.999f + 1.f;
}

// Writes one int: correctness probe for both argument ABIs.
KERNEL void hrx_store(unsigned* out, unsigned value) { out[0] = value; }
