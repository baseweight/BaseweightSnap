# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("baseweightsnap")

# Check if the target architecture is arm64-v8a
if(${CMAKE_ANDROID_ARCH_ABI} STREQUAL "arm64-v8a")
    # Set the C flags for arm64-v8a
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.4a+dotprod")
endif()

set(LLAMA_CURL OFF)

#load local llama.cpp
add_subdirectory(llama.cpp build-llama)
add_subdirectory(llama.cpp/common build-common)
add_subdirectory(llama.cpp/tools/mtmd build-mtmd)

add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        mtmd-android.cpp
        model_manager.cpp)

# Add include directories
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/common
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/tools/mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/src
        ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/src)

# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        llama
        common
        mtmd_shared
        android
        log)