# BaseweightSnap CMakeLists.txt
# Supports both Vulkan (built from source) and Hexagon (prebuilt) backends

cmake_minimum_required(VERSION 3.22.1)
project("baseweightsnap")

# =============================================================================
# Backend Selection
# =============================================================================
# Use -DBACKEND=vulkan or -DBACKEND=hexagon (default: vulkan)
set(BACKEND "vulkan" CACHE STRING "Backend to use: vulkan or hexagon")
set_property(CACHE BACKEND PROPERTY STRINGS vulkan hexagon)

message(STATUS "Selected backend: ${BACKEND}")

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

# =============================================================================
# Vulkan Backend (Built from source)
# =============================================================================
if(BACKEND STREQUAL "vulkan")
    message(STATUS "Building with Vulkan backend (from source)")
    
    set(GGML_VULKAN ON)
    set(GGML_CPU_KLEIDIAI OFF)
    
    # Vulkan headers
    set(VULKAN_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/Vulkan-Headers)
    set(VULKAN_HEADERS_INSTALL_DIR ${VULKAN_HEADERS_DIR})
    set(Vulkan_INCLUDE_DIR ${VULKAN_HEADERS_DIR}/include)
    include_directories(${VULKAN_HEADERS_DIR}/include)
    
    # Build llama.cpp from source
    add_subdirectory(llama.cpp build-llama)
    
    # Build mtmd from source
    set(LLAMA_INSTALL_VERSION "0.0.0")
    add_subdirectory(llama.cpp/tools/mtmd build-mtmd)
    
    # Source files
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

# =============================================================================
# Hexagon Backend (Prebuilt libraries from Snapdragon Docker)
# =============================================================================
elseif(BACKEND STREQUAL "hexagon")
    message(STATUS "Building with Hexagon backend (prebuilt libraries)")

    # Path to prebuilt Hexagon libraries
    # CMAKE_CURRENT_SOURCE_DIR = app/src/main/cpp, so ../../../ = app/
    set(HEXAGON_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../hexagon-libs/${ANDROID_ABI})

    # OpenCL headers (for Adreno GPU fallback)
    set(OPENCL_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/OpenCL-Headers)
    include_directories(${OPENCL_HEADERS_DIR})

    # Verify libraries exist
    if(NOT EXISTS ${HEXAGON_LIB_DIR}/libllama.so)
        message(FATAL_ERROR "Hexagon libraries not found at ${HEXAGON_LIB_DIR}\n"
                           "Copy prebuilt .so files to app/hexagon-libs/${ANDROID_ABI}/")
    endif()

    message(STATUS "Using Hexagon libraries from: ${HEXAGON_LIB_DIR}")

    # ---- Import prebuilt shared libraries ----

    add_library(llama SHARED IMPORTED)
    set_target_properties(llama PROPERTIES
        IMPORTED_LOCATION ${HEXAGON_LIB_DIR}/libllama.so
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include;${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include"
    )

    add_library(mtmd SHARED IMPORTED)
    set_target_properties(mtmd PROPERTIES
        IMPORTED_LOCATION ${HEXAGON_LIB_DIR}/libmtmd.so
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd"
    )

    add_library(ggml SHARED IMPORTED)
    set_target_properties(ggml PROPERTIES
        IMPORTED_LOCATION ${HEXAGON_LIB_DIR}/libggml.so
    )

    add_library(ggml-base SHARED IMPORTED)
    set_target_properties(ggml-base PROPERTIES
        IMPORTED_LOCATION ${HEXAGON_LIB_DIR}/libggml-base.so
    )

    # Backend libraries (ggml-cpu, ggml-opencl, ggml-hexagon, ggml-htp-v*)
    # are NOT imported as CMake targets. They are packaged into the APK
    # via jniLibs and loaded at runtime by llama's backend plugin system.

    # ---- Build 'common' static library from source ----
    # common is not in the prebuilt package (it's a static lib).
    # Our model_manager.cpp uses common_sampler_*, common_chat_templates_*, etc.

    # No-op stubs for functions defined in llama.cpp's top-level CMake
    function(llama_add_compile_flags)
    endfunction()
    function(license_add_file name file)
    endfunction()

    # cpp-httplib (vendored, required by common)
    add_subdirectory(llama.cpp/vendor/cpp-httplib build-httplib)

    # common static library
    add_subdirectory(llama.cpp/common build-common)

    # ---- Our native library ----
    add_library(baseweightsnap SHARED
            mtmd-android.cpp
            model_manager.cpp)

    target_include_directories(baseweightsnap PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common
            ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd
            ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/src
            ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include)

    # Backend libraries (ggml-hexagon, ggml-opencl, ggml-cpu, ggml-htp-v*)
    # are NOT linked directly. They are packaged into the APK via jniLibs
    # and loaded at runtime by llama's backend plugin system (dlopen).
    # Linking them here would cause transitive dependency failures
    # (e.g. libggml-opencl.so needs system libOpenCL.so which is
    # blocked by Android's linker namespace isolation).
    target_link_libraries(baseweightsnap
            llama
            common
            mtmd
            ggml
            ggml-base
            android
            log)

else()
    message(FATAL_ERROR "Unknown backend: ${BACKEND}. Use 'vulkan' or 'hexagon'.")
endif()
