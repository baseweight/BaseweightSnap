#!/bin/bash
# Hexagon Build Script for BaseweightSnap
# Run this on your host machine (not in Docker)

set -e

echo "================================================"
echo "Building BaseweightSnap with Hexagon NPU support"
echo "================================================"
echo ""

# Configuration
LLAMA_COMMIT="6a8cf8914"  # Pin to known-good commit
DOCKER_IMAGE="ghcr.io/snapdragon-toolchain/arm64-android:v0.3"
BUILD_DIR="app/src/main/cpp/llama.cpp"
OUTPUT_DIR="hexagon-libs"

# Check if llama.cpp submodule is at correct commit
cd "$BUILD_DIR"
CURRENT_COMMIT=$(git rev-parse --short HEAD)
if [ "$CURRENT_COMMIT" != "$LLAMA_COMMIT" ]; then
    echo "⚠️  Warning: llama.cpp is at $CURRENT_COMMIT, expected $LLAMA_COMMIT"
    echo "Checking out pinned commit..."
    git fetch origin
    git checkout "$LLAMA_COMMIT"
fi
cd ../../../../../

echo "✓ llama.cpp pinned to commit: $LLAMA_COMMIT"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🐳 Running Docker build with Snapdragon toolchain..."
echo "This will take 10-15 minutes..."
echo ""

# Run Docker build
docker run --rm \
    -u $(id -u):$(id -g) \
    -v $(pwd):/workspace \
    --platform linux/amd64 \
    "$DOCKER_IMAGE" \
    bash -c "
        set -e
        echo 'Inside Docker container...'
        cd /workspace/$BUILD_DIR
        
        # Copy Snapdragon preset
        echo 'Setting up CMake preset...'
        cp docs/backend/snapdragon/CMakeUserPresets.json .
        
        # Configure build with Hexagon + OpenCL + mtmd
        echo 'Configuring CMake...'
        cmake --preset arm64-android-snapdragon-release \
            -B build-hexagon \
            -DLLAMA_BUILD_TOOLS=ON \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_TESTS=OFF
        
        # Build everything
        echo 'Building llama.cpp with Hexagon...'
        cmake --build build-hexagon -j\$(nproc)
        
        # Install to output dir
        echo 'Installing libraries...'
        cmake --install build-hexagon \
            --prefix /workspace/$OUTPUT_DIR \
            --strip
        
        echo '✓ Build complete!'
    "

echo ""
echo "📦 Copying additional files..."

# Copy headers needed for compilation
cp -r "$BUILD_DIR/include" "$OUTPUT_DIR/" 2>/dev/null || true
cp -r "$BUILD_DIR/common" "$OUTPUT_DIR/include/" 2>/dev/null || true
cp -r "$BUILD_DIR/tools/mtmd" "$OUTPUT_DIR/include/" 2>/dev/null || true
cp -r "$BUILD_DIR/ggml/include" "$OUTPUT_DIR/" 2>/dev/null || true

# Create CMake config file for find_package
cat > "$OUTPUT_DIR/llama-config.cmake" << 'EOF'
# Config file for find_package(llama)
get_filename_component(LLAMA_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

set(LLAMA_INCLUDE_DIRS "${LLAMA_CMAKE_DIR}/include")
set(LLAMA_LIBRARIES "${LLAMA_CMAKE_DIR}/lib/libllama.so")

include("${LLAMA_CMAKE_DIR}/llama-targets.cmake" OPTIONAL)
EOF

# Create mtmd config
cat > "$OUTPUT_DIR/mtmd-config.cmake" << 'EOF'
get_filename_component(MTMD_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(MTMD_INCLUDE_DIRS "${MTMD_CMAKE_DIR}/include")
set(MTMD_LIBRARIES "${MTMD_CMAKE_DIR}/lib/libmtmd.so")
EOF

echo ""
echo "================================================"
echo "✅ Hexagon build complete!"
echo "================================================"
echo ""
echo "Built libraries:"
ls -la "$OUTPUT_DIR/lib/"*.so 2>/dev/null || echo "  (none found)"
echo ""
echo "Directory structure:"
echo "  hexagon-libs/"
echo "    ├── lib/          ← .so files (copy to jniLibs/arm64-v8a/)"
echo "    ├── include/      ← headers"
echo "    └── *.cmake       ← CMake configs"
echo ""
echo "Next steps:"
echo "  1. git checkout -b hexagon-build"
echo "  2. Update app/src/main/cpp/CMakeLists.txt"
echo "  3. Build APK: ./gradlew assembleRelease"
echo ""
