#!/bin/bash

# Script to build both CPU and Vulkan variants of the native library

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$SCRIPT_DIR/app/.cxx"
CMAKE_LISTS="$SCRIPT_DIR/app/src/main/cpp/CMakeLists.txt"

echo "================================"
echo "Building Native Library Variants"
echo "================================"

# Build CPU variant
echo ""
echo "Building CPU variant (NEON/KleidiAI)..."
echo "---------------------------------------"
cd "$SCRIPT_DIR"
./gradlew clean
./gradlew :app:externalNativeBuildDebug -DBUILD_VARIANT=cpu

# Copy CPU libraries
echo "Copying CPU variant libraries..."
mkdir -p app/src/main/jniLibs/arm64-v8a
mkdir -p app/src/main/jniLibs/x86_64
cp -v app/build/intermediates/cmake/debug/obj/arm64-v8a/libbaseweightsnap-cpu.so \
   app/src/main/jniLibs/arm64-v8a/ || true
cp -v app/build/intermediates/cmake/debug/obj/x86_64/libbaseweightsnap-cpu.so \
   app/src/main/jniLibs/x86_64/ || true

# Build Vulkan variant
echo ""
echo "Building Vulkan variant..."
echo "--------------------------"
./gradlew clean
./gradlew :app:externalNativeBuildDebug -DBUILD_VARIANT=vulkan

# Copy Vulkan libraries
echo "Copying Vulkan variant libraries..."
cp -v app/build/intermediates/cmake/debug/obj/arm64-v8a/libbaseweightsnap-vulkan.so \
   app/src/main/jniLibs/arm64-v8a/ || true
cp -v app/build/intermediates/cmake/debug/obj/x86_64/libbaseweightsnap-vulkan.so \
   app/src/main/jniLibs/x86_64/ || true

echo ""
echo "================================"
echo "Build Complete!"
echo "================================"
echo "CPU variant: libbaseweightsnap-cpu.so"
echo "Vulkan variant: libbaseweightsnap-vulkan.so"
echo ""
echo "Libraries copied to app/src/main/jniLibs/"
