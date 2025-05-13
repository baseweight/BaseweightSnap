# BaseWeightSnap

BaseWeight Snap is an Android application that atttempts to recreate the functionality of the the HuggingFace Snap application. 
The goal of this application is to show that it's possible to run Multimodal models on Android using llama.cpp and libmtmd.

## Features

- Real-time object detection and weight estimation using ONNX Runtime
- Camera integration using CameraX
- Native C++ processing for optimal performance
- Support for arm64-v8a 

## Technical Details

### Dependencies

- CameraX v1.3.2 for camera functionality
- llama.cpp with libmtmd for VLM functionality

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
- llama.cpp - symlinked into the `app/src/main/cpp` directory

## Building the Project

1. Clone the repository
2. Symlink the llama.cpp repository in `app/src/main/cpp`
3. Sync the project with Gradle files
4. Build and run the application.

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
