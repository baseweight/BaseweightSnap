#!/bin/bash

# Environment setup for static linking with ONNX Runtime 2.0.0-rc.10
# Source this file before building: source setup_static_env.sh

echo "Setting up environment for static ONNX Runtime linking (ort 2.0.0-rc.10)..."

# Path to ONNX Runtime static libraries (MinSizeRel build)
export ORT_LIB_LOCATION="/home/bowserj/ort/onnxruntime/build/Android/MinSizeRel"
export ORT_LIB_PROFILE="MinSizeRel"

# Set the strategy to use static linking
export ORT_STRATEGY="static"

# Android NDK environment
export ANDROID_NDK="/home/bowserj/Android/Sdk/ndk/28.0.12916984"

echo "✓ ORT_LIB_LOCATION: $ORT_LIB_LOCATION"
echo "✓ ORT_LIB_PROFILE: $ORT_LIB_PROFILE"
echo "✓ ORT_STRATEGY: $ORT_STRATEGY"
echo "✓ Environment ready for static linking build"
echo ""
echo "Now you can build with:"
echo "  cd smolvlm_rs_lib && cargo ndk -t arm64-v8a build --release"