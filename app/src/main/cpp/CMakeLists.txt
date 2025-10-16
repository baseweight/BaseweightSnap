# CMakeLists.txt for nanoVLM Android with ExecuTorch
cmake_minimum_required(VERSION 3.22.1)

project("baseweightsnap")

# Set C++ standard (required for ExecuTorch)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler definitions to match ExecuTorch build
add_definitions(-DC10_USING_CUSTOM_GENERATED_MACROS)
add_definitions(-DET_ENABLE_PROGRAM_VERIFICATION=0)
add_definitions(-DET_LOG_ENABLED=0)
add_definitions(-DET_MIN_LOG_LEVEL=Info)

# Check if the target architecture is arm64-v8a
if(${CMAKE_ANDROID_ARCH_ABI} STREQUAL "arm64-v8a")
    # Set the C flags for arm64-v8a
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.4a+dotprod")
endif()

# ExecuTorch configuration
set(EXECUTORCH_ROOT "/home/bowserj/executorch" CACHE PATH "Path to ExecuTorch root directory")
set(EXECUTORCH_BUILD_DIR "/home/bowserj/executorch/cmake-out-android-arm64" CACHE PATH "Path to ExecuTorch Android build")

message(STATUS "ExecuTorch root: ${EXECUTORCH_ROOT}")
message(STATUS "ExecuTorch build: ${EXECUTORCH_BUILD_DIR}")

# ExecuTorch include directories (all the paths needed)
include_directories(
    ${EXECUTORCH_ROOT}/..
    ${EXECUTORCH_ROOT}
    ${EXECUTORCH_ROOT}/runtime/core
    ${EXECUTORCH_ROOT}/runtime/core/exec_aten
    ${EXECUTORCH_ROOT}/runtime/core/portable_type
    ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
    ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10/c10
    ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10/c10/macros
    ${EXECUTORCH_ROOT}/extension/module
    ${EXECUTORCH_ROOT}/extension/tensor
    ${EXECUTORCH_BUILD_DIR}
)

# Rust tokenizer include (relative to cpp directory)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../rust-tokenizer/include
)

# nlohmann/json include
include_directories(
    ${EXECUTORCH_ROOT}/third-party/json/single_include
)

# ExecuTorch libraries
add_library(executorch STATIC IMPORTED)
set_target_properties(executorch PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/libexecutorch.a
)

add_library(executorch_core STATIC IMPORTED)
set_target_properties(executorch_core PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/libexecutorch_core.a
)

add_library(extension_module_static STATIC IMPORTED)
set_target_properties(extension_module_static PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/extension/module/libextension_module_static.a
)

add_library(extension_tensor STATIC IMPORTED)
set_target_properties(extension_tensor PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/extension/tensor/libextension_tensor.a
)

add_library(portable_ops_lib STATIC IMPORTED)
set_target_properties(portable_ops_lib PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/kernels/portable/libportable_ops_lib.a
)

add_library(portable_kernels STATIC IMPORTED)
set_target_properties(portable_kernels PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/kernels/portable/libportable_kernels.a
)

add_library(custom_ops STATIC IMPORTED)
set_target_properties(custom_ops PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/extension/llm/custom_ops/libcustom_ops.a
)

# Optimized ops (includes SDPA and other optimized kernels)
add_library(optimized_native_cpu_ops_lib STATIC IMPORTED)
set_target_properties(optimized_native_cpu_ops_lib PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/configurations/liboptimized_native_cpu_ops_lib.a
)

# Optimized kernels (implementations)
add_library(optimized_kernels STATIC IMPORTED)
set_target_properties(optimized_kernels PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/kernels/optimized/liboptimized_kernels.a
)

# CPU BLAS for optimized ops
add_library(cpublas STATIC IMPORTED)
set_target_properties(cpublas PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/kernels/optimized/libcpublas.a
)

# Eigen BLAS for optimized ops
add_library(eigen_blas STATIC IMPORTED)
set_target_properties(eigen_blas PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/kernels/optimized/libeigen_blas.a
)

# XNNPACK backend libraries
add_library(xnnpack_backend STATIC IMPORTED)
set_target_properties(xnnpack_backend PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/backends/xnnpack/libxnnpack_backend.a
)

add_library(XNNPACK STATIC IMPORTED)
set_target_properties(XNNPACK PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/backends/xnnpack/third-party/XNNPACK/libXNNPACK.a
)

add_library(xnnpack_microkernels STATIC IMPORTED)
set_target_properties(xnnpack_microkernels PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/backends/xnnpack/third-party/XNNPACK/libxnnpack-microkernels-prod.a
)

# XNNPACK dependencies
add_library(cpuinfo STATIC IMPORTED)
set_target_properties(cpuinfo PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/backends/xnnpack/third-party/cpuinfo/libcpuinfo.a
)

add_library(pthreadpool STATIC IMPORTED)
set_target_properties(pthreadpool PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/backends/xnnpack/third-party/pthreadpool/libpthreadpool.a
)

# KleidiAI library (for XNNPACK quantized operations)
add_library(kleidiai STATIC IMPORTED)
set_target_properties(kleidiai PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/kleidiai/libkleidiai.a
)

# ExecuTorch threadpool extension (for SDPA custom ops)
add_library(extension_threadpool STATIC IMPORTED)
set_target_properties(extension_threadpool PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/extension/threadpool/libextension_threadpool.a
)

# Data loader libraries
add_library(extension_data_loader STATIC IMPORTED)
set_target_properties(extension_data_loader PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/extension/data_loader/libextension_data_loader.a
)

# Flat tensor libraries
add_library(extension_flat_tensor STATIC IMPORTED)
set_target_properties(extension_flat_tensor PROPERTIES
    IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/extension/flat_tensor/libextension_flat_tensor.a
)

# Note: flatcc libraries are built for x86-64, not aarch64, so we skip them
# add_library(flatcc STATIC IMPORTED)
# set_target_properties(flatcc PROPERTIES
#     IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/third-party/flatcc_external_project/lib/libflatcc.a
# )

# add_library(flatccrt STATIC IMPORTED)
# set_target_properties(flatccrt PROPERTIES
#     IMPORTED_LOCATION ${EXECUTORCH_BUILD_DIR}/third-party/flatcc_external_project/lib/libflatccrt.a
# )

# Rust tokenizer library (will be built for Android)
add_library(nanovlm_preprocessor STATIC IMPORTED)
set_target_properties(nanovlm_preprocessor PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../../../rust-tokenizer/target/aarch64-linux-android/release/libnanovlm_preprocessor.a
)

# Main shared library
add_library(${CMAKE_PROJECT_NAME} SHARED
    nanovlm_android.cpp
    image_preprocessor.cpp
)

# Include directories for our code
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
    # ExecuTorch core
    executorch
    executorch_core
    extension_module_static
    extension_tensor

    # Data loaders and flat tensor support
    extension_data_loader
    extension_flat_tensor
    # flatcc and flatccrt are built for x86-64, not aarch64

    # Operators - XNNPACK backend + custom ops (including SDPA)
    # Use --whole-archive to ensure operator registration symbols are not stripped
    -Wl,--whole-archive
    xnnpack_backend
    portable_ops_lib
    portable_kernels
    custom_ops
    -Wl,--no-whole-archive

    # XNNPACK core library and dependencies (don't need whole-archive)
    XNNPACK
    xnnpack_microkernels
    kleidiai
    cpuinfo
    pthreadpool

    # Optimized BLAS for SDPA custom ops
    cpublas
    eigen_blas
    extension_threadpool

    # Rust tokenizer
    nanovlm_preprocessor

    # Android system libraries
    android
    log
)

message(STATUS "Configured nanoVLM Android with ExecuTorch (portable ops)")
