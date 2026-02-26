import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

// Load local.properties for signing configuration
val localProperties = Properties()
val localPropertiesFile = rootProject.file("local.properties")
if (localPropertiesFile.exists()) {
    localPropertiesFile.inputStream().use { localProperties.load(it) }
}

android {
    namespace = "ai.baseweight.baseweightsnap"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.baseweight.baseweightsnap"
        minSdk = 28
        targetSdk = 35
        versionCode = 10
        versionName = "1.5.1"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                arguments(
                    "-DANDROID_STL=c++_shared"
                )
            }
        }

    }

    signingConfigs {
        create("release") {
            storeFile = file("/home/bowserj/baseweight_secrets/baseweight_snap_upload.jks")
            storePassword = localProperties.getProperty("BASEWEIGHT_KEYSTORE_PASSWORD") ?: ""
            keyAlias = "upload"
            keyPassword = localProperties.getProperty("BASEWEIGHT_KEY_PASSWORD") ?: ""
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            signingConfig = signingConfigs.getByName("release")
        }
    }

    // Backend variants: Vulkan (from source) or Hexagon (prebuilt)
    flavorDimensions += "backend"
    productFlavors {
        create("vulkan") {
            dimension = "backend"
            ndk {
                abiFilters.addAll(listOf("arm64-v8a", "x86_64"))
            }
            externalNativeBuild {
                cmake {
                    arguments("-DBACKEND=vulkan")
                }
            }
        }
        create("hexagon") {
            dimension = "backend"
            externalNativeBuild {
                cmake {
                    arguments("-DBACKEND=hexagon")
                }
            }
            // Hexagon is Qualcomm-only, no x86_64
            ndk {
                abiFilters.clear()
                abiFilters.add("arm64-v8a")
            }
            // Package prebuilt Hexagon/OpenCL/HTP .so files into APK
            sourceSets.getByName("hexagon") {
                jniLibs.srcDirs("hexagon-libs")
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    // Configure native builds for multiple library variants
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }
    buildFeatures {
        viewBinding = true
    }
    ndkVersion = project.findProperty("android.ndkVersion")?.toString() ?: "27.0.12077973"
    
    packaging {
        jniLibs {
            pickFirsts.add("lib/arm64-v8a/libc++_shared.so")
            pickFirsts.add("lib/x86_64/libc++_shared.so")
            useLegacyPackaging = true
        }
    }
}

dependencies {

    // CameraX
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    implementation(libs.androidx.camera.extensions)

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)

    // Testing
    testImplementation(libs.junit)
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
    testImplementation("io.mockk:mockk:1.13.8")
    testImplementation("androidx.arch.core:core-testing:2.2.0")
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation("androidx.test:runner:1.5.2")
    androidTestImplementation("androidx.test:rules:1.5.0")

    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // JSON serialization for model metadata
    implementation("com.google.code.gson:gson:2.10.1")
}