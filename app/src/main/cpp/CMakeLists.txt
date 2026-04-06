# BaseweightSnap CMakeLists.txt
# Unified build: Vulkan + OpenCL from source, Hexagon prebuilt via jniLibs

cmake_minimum_required(VERSION 3.22.1)
project("baseweightsnap")

# =============================================================================
# Common Settings
# =============================================================================
set(LLAMA_CURL OFF)
set(BUILD_NUMBER 0)
set(BUILD_COMMIT "unknown")
set(BUILD_COMPILER "unknown")
set(BUILD_TARGET "unknown")
set(LLAMA_BUILD_NUMBER 0)
set(LLAMA_BUILD_COMMIT "unknown")
set(LLAMA_BUILD_COMMON ON)
set(LLAMA_BUILD_TOOLS OFF)

set(GGML_CPU_KLEIDIAI OFF)

# =============================================================================
# Vulkan Backend (from source)
# =============================================================================
set(GGML_VULKAN ON)

set(VULKAN_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/Vulkan-Headers)
set(VULKAN_HEADERS_INSTALL_DIR ${VULKAN_HEADERS_DIR})
set(Vulkan_INCLUDE_DIR ${VULKAN_HEADERS_DIR}/include)
include_directories(${VULKAN_HEADERS_DIR}/include)

# =============================================================================
# OpenCL Backend (from source, ARM64 only — for Qualcomm Adreno GPUs)
# =============================================================================
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_OPENCL ON)
    set(GGML_OPENCL_EMBED_KERNELS ON)
    set(GGML_OPENCL_USE_ADRENO_KERNELS ON)

    # Custom FindOpenCL.cmake for Android cross-compilation
    # Headers from third_party/OpenCL-Headers, stub lib from device for link-time
    set(OPENCL_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/OpenCL-Headers)
    set(OPENCL_STUB_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/opencl-stub/libOpenCL.so)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
endif()

# =============================================================================
# Build llama.cpp from source (includes ggml with Vulkan + OpenCL backends)
# =============================================================================
add_subdirectory(llama.cpp build-llama)

# Build mtmd from source
set(LLAMA_INSTALL_VERSION "0.0.0")
add_subdirectory(llama.cpp/tools/mtmd build-mtmd)

# =============================================================================
# Our native library
# =============================================================================
add_library(baseweightsnap SHARED
        mtmd-android.cpp
        model_manager.cpp)

target_include_directories(baseweightsnap PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/src
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/src)

target_link_libraries(baseweightsnap
        llama
        common
        mtmd
        android
        log)
