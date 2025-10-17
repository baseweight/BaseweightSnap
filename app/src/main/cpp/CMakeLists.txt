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
set(LLAMA_BUILD_TOOLS ON)

# Allow selection of build variant via BUILD_VARIANT option
# Default to CPU variant if not specified
if(NOT DEFINED BUILD_VARIANT)
    set(BUILD_VARIANT "cpu")
endif()

message(STATUS "Building variant: ${BUILD_VARIANT}")

if(BUILD_VARIANT STREQUAL "cpu")
    # =============================================================================
    # CPU Variant (NEON/KleidiAI optimized)
    # =============================================================================

    # Check if the target architecture is arm64-v8a
    if(${CMAKE_ANDROID_ARCH_ABI} STREQUAL "arm64-v8a")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.4a+dotprod")
        set(GGML_CPU_KLEIDIAI ON)
        set(FETCHCONTENT_SOURCE_DIR_KLEIDIAI_DOWNLOAD /home/bowserj/kleidiai)
    else()
        set(GGML_CPU_KLEIDIAI OFF)
    endif()

    set(GGML_VULKAN OFF)

    # Load llama.cpp for CPU variant
    add_subdirectory(llama.cpp build-llama)

    # Build CPU variant library
    add_library(baseweightsnap-cpu SHARED
            mtmd-android.cpp
            model_manager.cpp)

    target_include_directories(baseweightsnap-cpu PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common
            ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd
            ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/src
            ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/src)

    target_link_libraries(baseweightsnap-cpu
            llama
            common
            mtmd
            android
            log)

elseif(BUILD_VARIANT STREQUAL "vulkan")
    # =============================================================================
    # Vulkan Variant
    # =============================================================================

    set(GGML_VULKAN ON)
    set(GGML_CPU_KLEIDIAI OFF)

    # Load llama.cpp for Vulkan variant
    add_subdirectory(llama.cpp build-llama)

    # Build Vulkan variant library
    add_library(baseweightsnap-vulkan SHARED
            mtmd-android.cpp
            model_manager.cpp)

    target_include_directories(baseweightsnap-vulkan PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common
            ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd
            ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/src
            ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/src)

    target_link_libraries(baseweightsnap-vulkan
            llama
            common
            mtmd
            android
            log)

else()
    message(FATAL_ERROR "Invalid BUILD_VARIANT: ${BUILD_VARIANT}. Must be 'cpu' or 'vulkan'")
endif()