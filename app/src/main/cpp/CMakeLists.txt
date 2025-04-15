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

set(OpenCV_Android ${CMAKE_SOURCE_DIR}/../../../../../external/OpenCV-android-sdk)
set(rapidjson ${CMAKE_SOURCE_DIR}/../../../../../external/rapidjson)

# ONNX Runtime configuration
set(ORT_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/ort_sdk/include")
set(ORT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/ort_sdk/libs/${ANDROID_ABI}")

# Check if ONNX Runtime library exists
if(NOT EXISTS "${ORT_LIB_DIR}/libonnxruntime.so")
    message(FATAL_ERROR "ONNX Runtime library not found at '${ORT_LIB_DIR}/libonnxruntime.so'! Make sure the extract-ort.gradle.kts script has been executed.")
endif()

# Import ONNX Runtime
add_library(onnxruntime SHARED IMPORTED)
set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION "${ORT_LIB_DIR}/libonnxruntime.so")

# Bring in OpenCV from an external directory
add_library(lib_opencv SHARED IMPORTED)

set_target_properties(
        lib_opencv
        PROPERTIES
        IMPORTED_LOCATION ${OpenCV_Android}/sdk/native/libs/${ANDROID_ABI}/libopencv_java4.so)

include_directories(
        ${ORT_INCLUDE_DIRS}
        ${OpenCV_Android}/sdk/native/jni/include
        ${rapidjson}/include)

link_directories(
        ${ORT_LIB_DIR}
        ${OpenCV_Android}/sdk/native/libs/${ANDROID_ABI})

add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        ImageProcessor.cpp
        SmolVLMTokenizer.cpp
        SmolVLM.cpp
        native-lib.cpp)

# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        android
        lib_opencv
        onnxruntime
        log
        c++_shared)  # Add c++_shared library