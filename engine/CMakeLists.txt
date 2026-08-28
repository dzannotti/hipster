cmake_minimum_required(VERSION 3.21)
project(hipster LANGUAGES CXX HIP)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_ARCHITECTURES gfx1151)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_HIP_FLAGS_RELEASE "-O3")
find_package(hip REQUIRED)

add_library(hipster_core STATIC src/gguf.cpp kernels/gemv.hip)
target_include_directories(hipster_core PUBLIC src kernels)
target_link_libraries(hipster_core PUBLIC hip::host)

add_executable(bench_gemv bench/bench_gemv.cpp)
target_link_libraries(bench_gemv hipster_core)

target_sources(hipster_core PRIVATE kernels/ops.hip kernels/gemm.hip kernels/fn.hip src/qwen35.cpp src/qwen4exp.cpp)
target_link_directories(hipster_core PUBLIC /opt/rocm/lib)
target_link_libraries(hipster_core PUBLIC /opt/rocm/lib/libhipblaslt.so)
add_executable(run27b bench/run27b.cpp)
target_link_libraries(run27b hipster_core)

add_executable(spec27b bench/spec27b.cpp)
target_link_libraries(spec27b hipster_core)

add_executable(prefill27b bench/prefill27b.cpp)
target_link_libraries(prefill27b hipster_core)

add_executable(runfn bench/runfn.cpp)
target_link_libraries(runfn hipster_core)

add_executable(specfn bench/specfn.cpp)
target_link_libraries(specfn hipster_core)

add_executable(batchfn bench/batchfn.cpp)
target_link_libraries(batchfn hipster_core)
