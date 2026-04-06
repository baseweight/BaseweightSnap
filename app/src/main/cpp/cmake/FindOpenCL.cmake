# FindOpenCL.cmake - Android cross-compilation stub
#
# OpenCL headers: third_party/OpenCL-Headers
# Link-time stub: third_party/opencl-stub/libOpenCL.so (pulled from device)
# Runtime library: /vendor/lib64/libOpenCL.so on the device
#
# The caller must set OPENCL_HEADERS_DIR and OPENCL_STUB_LIB before
# this module is found via CMAKE_MODULE_PATH.

if(NOT OPENCL_HEADERS_DIR OR NOT OPENCL_STUB_LIB)
    message(FATAL_ERROR "Set OPENCL_HEADERS_DIR and OPENCL_STUB_LIB before find_package(OpenCL)")
endif()

set(OpenCL_FOUND TRUE)
set(OpenCL_INCLUDE_DIRS "${OPENCL_HEADERS_DIR}")
set(OpenCL_LIBRARIES "${OPENCL_STUB_LIB}")

if(NOT TARGET OpenCL::OpenCL)
    add_library(OpenCL::OpenCL SHARED IMPORTED)
    set_target_properties(OpenCL::OpenCL PROPERTIES
        IMPORTED_LOCATION "${OPENCL_STUB_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENCL_HEADERS_DIR}"
    )
endif()
