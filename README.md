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

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Contributing

We welcome contributions to BaseWeightSnap! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code quality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure your PR:
- Follows the existing code style
- Includes appropriate tests
- Updates documentation as needed
- Describes the changes made

For major changes, please open an issue first to discuss what you would like to change.
