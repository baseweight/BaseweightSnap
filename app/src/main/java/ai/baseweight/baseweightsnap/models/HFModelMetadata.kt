package ai.baseweight.baseweightsnap.models

import java.util.UUID

/**
 * Metadata for a downloaded HuggingFace model
 */
data class HFModelMetadata(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val hfRepo: String,              // "orgName/repository"
    val languageFile: String,        // Filename of main GGUF
    val visionFile: String,          // Filename of mmproj GGUF
    val configFile: String = "config.json",
    val downloadDate: Long = System.currentTimeMillis(),
    val isDefault: Boolean = false,
    val languageSize: Long,
    val visionSize: Long
) {
    val totalSize: Long get() = languageSize + visionSize

    fun formatSize(): String {
        val totalMB = totalSize / (1024 * 1024)
        return if (totalMB < 1024) {
            "$totalMB MB"
        } else {
            String.format("%.1f GB", totalMB / 1024.0)
        }
    }
}

/**
 * Container for storing all model metadata
 */
data class ModelMetadataStore(
    val models: List<HFModelMetadata> = emptyList(),
    val defaultModelId: String? = null
)

/**
 * HuggingFace repository information
 */
data class HFRepoInfo(
    val id: String,
    val author: String,
    val modelId: String,
    val sha: String?,
    val lastModified: String?,
    val private: Boolean = false,
    val gated: Boolean = false,
    val downloads: Long = 0
)

/**
 * Represents a file in a HuggingFace repository
 */
data class HFFile(
    val path: String,
    val size: Long,
    val type: String = "file"  // "file" or "directory"
) {
    val isGGUF: Boolean get() = path.endsWith(".gguf", ignoreCase = true)
    val isMMProj: Boolean get() = path.contains("mmproj", ignoreCase = true) && isGGUF
    val isConfig: Boolean get() = path.equals("config.json", ignoreCase = true)
}

/**
 * Required files for a VLM model
 */
data class HFModelFiles(
    val configFile: HFFile,
    val languageFile: HFFile,
    val visionFile: HFFile
) {
    val totalSize: Long get() = configFile.size + languageFile.size + visionFile.size
}

/**
 * Result of validating a HuggingFace repository
 */
sealed class ValidationResult {
    data class Valid(val files: HFModelFiles) : ValidationResult()
    data class Invalid(val error: ValidationError) : ValidationResult()
}

enum class ValidationError {
    REPO_NOT_FOUND,
    MISSING_CONFIG,
    MISSING_LANGUAGE_MODEL,
    MISSING_VISION_MODEL,
    NETWORK_ERROR,
    INVALID_REPO_FORMAT,
    MULTIPLE_LANGUAGE_MODELS,
    MULTIPLE_VISION_MODELS
}
