#!/bin/bash
# Build the Hexagon NPU dynamic ggml backend for BaseweightSnap.
#
# The unified APK builds the Vulkan and OpenCL backends statically from the
# llama.cpp submodule (see app/src/main/cpp/CMakeLists.txt). The Hexagon
# backend needs Qualcomm's Snapdragon toolchain, so it is built here inside
# their Docker image and shipped as prebuilt dynamic backend libraries in
# app/hexagon-only-libs/ (gitignored — rerun this script after cloning or
# after bumping the llama.cpp submodule).
#
# The dynamically loaded backend must match the statically linked ggml, so
# this script refuses to build unless the submodule checkout matches the
# commit recorded in the superproject.

set -e
cd "$(dirname "$0")/.."

DOCKER_IMAGE="ghcr.io/snapdragon-toolchain/arm64-android:v0.3"
SUBMODULE="app/src/main/cpp/llama.cpp"
OUTPUT_DIR="hexagon-libs"
JNILIBS_DIR="app/hexagon-only-libs/arm64-v8a"

echo "================================================"
echo "Building Hexagon NPU backend for BaseweightSnap"
echo "================================================"
echo ""

# The prebuilt backend is only ABI-compatible with the exact ggml it was
# built from, so enforce the submodule commit recorded in the superproject.
EXPECTED_COMMIT=$(git ls-tree HEAD "$SUBMODULE" | awk '{print $3}')
CURRENT_COMMIT=$(git -C "$SUBMODULE" rev-parse HEAD)
if [ "$EXPECTED_COMMIT" != "$CURRENT_COMMIT" ]; then
    echo "❌ llama.cpp submodule is at $CURRENT_COMMIT"
    echo "   but the superproject records $EXPECTED_COMMIT."
    echo "   Run: git submodule update --init $SUBMODULE"
    exit 1
fi
echo "✓ llama.cpp submodule at recorded commit: $CURRENT_COMMIT"
echo ""

# Snapdragon CMake preset (tracked in-repo so it survives submodule updates)
cp scripts/snapdragon/CMakeUserPresets.json "$SUBMODULE/CMakeUserPresets.json"

mkdir -p "$OUTPUT_DIR"

echo "🐳 Running Docker build with Snapdragon toolchain..."
echo "This will take 10-15 minutes..."
echo ""

docker run --rm \
    -u $(id -u):$(id -g) \
    -v $(pwd):/workspace \
    --platform linux/amd64 \
    "$DOCKER_IMAGE" \
    bash -c "
        set -e
        cd /workspace/$SUBMODULE

        echo 'Configuring CMake...'
        cmake --preset arm64-android-snapdragon-release \
            -B build-hexagon \
            -DLLAMA_CURL=OFF \
            -DLLAMA_BUILD_TOOLS=OFF \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_TESTS=OFF

        echo 'Building ggml with Hexagon backend...'
        cmake --build build-hexagon -j\$(nproc)

        echo 'Installing libraries...'
        cmake --install build-hexagon \
            --prefix /workspace/$OUTPUT_DIR \
            --strip
    "

echo ""
echo "📦 Copying backend libraries into $JNILIBS_DIR ..."
mkdir -p "$JNILIBS_DIR"

# libggml-hexagon.so — the dynamic backend, loaded via ggml_backend_load().
#   Its libggml-base.so dependency is NOT copied: the APK already ships the
#   one built by the app's own externalNativeBuild from the same submodule
#   commit, and the dynamic linker resolves against it (single ggml registry).
# libggml-htp-v*.so — Hexagon DSP skels, found via ADSP_LIBRARY_PATH
copied=0
for lib in libggml-hexagon.so; do
    src=$(find "$OUTPUT_DIR/bin" "$OUTPUT_DIR/lib" -name "$lib" 2>/dev/null | head -1)
    if [ -z "$src" ]; then
        echo "❌ $lib not found in $OUTPUT_DIR — build output layout changed?"
        exit 1
    fi
    cp "$src" "$JNILIBS_DIR/"
    copied=$((copied + 1))
done
for src in "$OUTPUT_DIR"/lib/libggml-htp-v*.so; do
    cp "$src" "$JNILIBS_DIR/"
    copied=$((copied + 1))
done

echo ""
echo "================================================"
echo "✅ Hexagon backend build complete ($copied libs)"
echo "================================================"
ls -la "$JNILIBS_DIR"
echo ""
echo "These are packaged into the APK via jniLibs.srcDirs (app/build.gradle.kts)."
echo "Build the APK as usual: ./gradlew assembleRelease"
