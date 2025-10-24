plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "ai.baseweight.baseweightsnap"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.baseweight.baseweightsnap"
        minSdk = 28
        targetSdk = 35
        versionCode = 8
        versionName = "1.4.1"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                arguments(
                    "-DANDROID_STL=c++_shared",
                    "-DBUILD_VARIANT=${project.findProperty("BUILD_VARIANT") ?: "cpu"}"
                )
            }
            // Don't build 32 bit libraries in 2025
            ndk {
                abiFilters.add("arm64-v8a")
                abiFilters.add("x86_64")
            }
        }

    }

    signingConfigs {
        create("release") {
            storeFile = file("/home/bowserj/baseweight_secrets/baseweight_snap_upload.jks")
            storePassword = System.getenv("BASEWEIGHT_KEYSTORE_PASSWORD") ?: ""
            keyAlias = "upload"
            keyPassword = System.getenv("BASEWEIGHT_KEY_PASSWORD") ?: ""
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
    ndkVersion = "28.2.13676358"
    
    packaging {
        jniLibs {
            pickFirsts.add("lib/arm64-v8a/libc++_shared.so")
            pickFirsts.add("lib/x86_64/libc++_shared.so")
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
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // JSON serialization for model metadata
    implementation("com.google.code.gson:gson:2.10.1")
}