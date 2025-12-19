# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name
project("baseweightsnap")

# Common settings
set(LLAMA_CURL OFF)
set(BUILD_NUMBER 0)
set(BUILD_COMMIT "unknown")
set(BUILD_COMPILER "unknown")
set(BUILD_TARGET "unknown")
set(LLAMA_BUILD_NUMBER 0)
set(LLAMA_BUILD_COMMIT "unknown")
set(LLAMA_BUILD_COMMON ON)
set(LLAMA_BUILD_TOOLS OFF)

# =============================================================================
# Vulkan Variant (Simplified - only build option)
# =============================================================================

set(GGML_VULKAN ON)
set(GGML_CPU_KLEIDIAI OFF)

# Set Vulkan headers path (relative to project root)
set(VULKAN_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/Vulkan-Headers)
set(VULKAN_HEADERS_INSTALL_DIR ${VULKAN_HEADERS_DIR})
set(Vulkan_INCLUDE_DIR ${VULKAN_HEADERS_DIR}/include)
include_directories(${VULKAN_HEADERS_DIR}/include)

# Load llama.cpp
add_subdirectory(llama.cpp build-llama)

# Build mtmd (multimodal) library - not built when LLAMA_BUILD_TOOLS is OFF
set(LLAMA_INSTALL_VERSION "0.0.0")
add_subdirectory(llama.cpp/tools/mtmd build-mtmd)

# Build Vulkan library
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