#!/bin/bash

# Build script for static linking with ONNX Runtime
# This script sets the proper environment variables for static linking

# Path to ONNX Runtime static libraries (MinSizeRel build)
export ORT_LIB_LOCATION="/home/bowserj/ort/onnxruntime/build/Android/MinSizeRel"
export ORT_LIB_PROFILE="MinSizeRel"

# Set the strategy to use static linking (instead of dynamic linking)
export ORT_STRATEGY="static"

# Android NDK environment
export ANDROID_NDK="/home/bowserj/Android/Sdk/ndk/28.0.12916984"

echo "Building with static ONNX Runtime linking..."
echo "ORT_LIB_LOCATION: $ORT_LIB_LOCATION"
echo "ORT_LIB_PROFILE: $ORT_LIB_PROFILE"
echo "ORT_STRATEGY: $ORT_STRATEGY"

cd smolvlm_rs_lib

# Build for Android ARM64
cargo ndk -t arm64-v8a build --release

echo "Build completed!"