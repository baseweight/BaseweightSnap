# BaseWeightSnap

BaseWeight Snap is an Android application that matches the HuggingFace Snap application.  The goal of this application
is to demonstrate how to pull down models and the encryption keys from Baseweight Park such that the models can be decrypted
and run on an Android device.

## Features

- Real-time object detection and weight estimation using ONNX Runtime
- Camera integration using CameraX
- Native C++ processing for optimal performance
- Support for arm64-v8a and x86_64 architectures

## Technical Details

### Dependencies

- ONNX Runtime Mobile v1.21.0 for ML model inference
- OpenCV for image processing
- CameraX v1.3.2 for camera functionality
- RapidJSON for data handling

### Build Requirements

- Android Studio
- CMake 3.22.1
- NDK version 28.0.12916984 rc3
- Minimum SDK: 28
- Target SDK: 35

### Project Structure

- `app/src/main/cpp/` - Native C++ code
- `app/src/main/java/` - Kotlin/Java source code
- External dependencies:
  - OpenCV Android SDK
  - RapidJSON
  - ONNX Runtime

## Building the Project

1. Clone the repository
2. Place the OpenCV Android SDK in the `external/OpenCV-android-sdk` directory
3. Place RapidJSON in the `external/rapidjson` directory
4. Sync the project with Gradle files
5. Build and run the application

## License

[Your license information here]

## Contributing

[Your contribution guidelines here]
