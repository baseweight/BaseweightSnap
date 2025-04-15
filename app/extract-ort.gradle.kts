configurations {
    create("onnxRuntimeDependency") {
        if (configurations.findByName("implementation") != null) {
            extendsFrom(configurations.getByName("implementation"))
        }
        isCanBeResolved = true
    }
}

val LIBRARY_NAME = "onnxruntime-android"
val LIBRARY_VERSION = "1.21.0" // Explicitly define version

// Copy the AAR file to the build directory
tasks.register<Copy>("copyAarToBuildDir") {
    from(configurations.getByName("onnxRuntimeDependency"))
    include("**/$LIBRARY_NAME*.aar")
    // Keep the version suffix in the copy to ensure we're using the correct file
    into("$buildDir/$LIBRARY_NAME")
    doLast {
        println("copyAarToBuildDir: copied AAR to $buildDir/$LIBRARY_NAME")
    }
}

// Extract the AAR content
tasks.register<Copy>("extractAarContent") {
    dependsOn("copyAarToBuildDir")
    from(zipTree("$buildDir/$LIBRARY_NAME/${LIBRARY_NAME}-${LIBRARY_VERSION}.aar"))
    into("$buildDir/$LIBRARY_NAME/content")
    doLast {
        println("extractAarContent: extracted AAR to $buildDir/$LIBRARY_NAME/content")
    }
}

// Copy C++ headers to project build directory
tasks.register<Copy>("copyHeadersToProjectBuildDir") {
    dependsOn("extractAarContent")
    from("$buildDir/$LIBRARY_NAME/content/headers")
    into("$projectDir/build/ort_sdk/include")
    doLast {
        println("copyHeadersToProjectBuildDir: copied headers to $projectDir/build/ort_sdk/include")
    }
}

// Copy JNI libraries to project build directory
tasks.register<Copy>("copyJniLibsToProjectBuildDir") {
    dependsOn("extractAarContent")
    from("$buildDir/$LIBRARY_NAME/content/jni")
    // Make sure all architectures are properly preserved
    includeEmptyDirs = true
    into("$projectDir/build/ort_sdk/libs")
    doLast {
        println("copyJniLibsToProjectBuildDir: copied JNI libs to $projectDir/build/ort_sdk/libs")
    }
}

// Make all required tasks run during preBuild
tasks.named("preBuild").configure {
    dependsOn("copyHeadersToProjectBuildDir", "copyJniLibsToProjectBuildDir")
}

tasks.named("clean").configure {
    doLast {
        delete("$projectDir/build/ort_sdk")
    }
}