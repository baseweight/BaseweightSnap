package ai.baseweight.baseweightsnap.models

import java.util.UUID

/**
 * Download state for a model
 */
enum class DownloadState {
    PENDING,      // Waiting to start
    DOWNLOADING,  // Actively downloading
    COMPLETED,    // Download finished
    ERROR         // Download failed
}

/**
 * Metadata for a downloaded HuggingFace model
 */
data class HFModelMetadata(
    val id: String = UUID.randomUUID().toString(),
    val name: String,
    val hfRepo: String,              // "orgName/repository"
    val languageFile: String,        // Filename of main GGUF
    val visionFile: String,          // Filename of mmproj GGUF (may be same as languageFile for unified models)
    val configFile: String? = null, // config.json not needed for GGUF models
    val downloadDate: Long = System.currentTimeMillis(),
    val isDefault: Boolean = false,
    val languageSize: Long,
    val visionSize: Long,
    val downloadState: DownloadState? = DownloadState.COMPLETED,  // Nullable for backwards compatibility
    val downloadProgress: Int = 100,  // 0-100
    val isUnified: Boolean = false    // NEW: true if languageFile == visionFile (unified GGUF like Gemma 3)
) {
    val totalSize: Long get() = if (isUnified) languageSize else languageSize + visionSize

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
 * HuggingFace Manifest response (from v2 API with llama-cpp User-Agent)
 * Used to detect unified GGUF files
 */
data class HFManifest(
    val ggufFile: String,
    val mmprojFile: String?
) {
    fun isUnified(): Boolean = ggufFile == mmprojFile && !mmprojFile.isNullOrEmpty()
    fun hasVision(): Boolean = !mmprojFile.isNullOrEmpty()
}

/**
 * Required files for a VLM model
 */
data class HFModelFiles(
    val configFile: HFFile?, // config.json not needed for GGUF models
    val languageFile: HFFile,
    val visionFile: HFFile?,  // Nullable for text-only models or will point to same file for unified
    val isUnified: Boolean = false  // True if languageFile == visionFile (unified GGUF like Gemma 3)
) {
    val totalSize: Long get() = (configFile?.size ?: 0) + languageFile.size + (if (isUnified) 0 else (visionFile?.size ?: 0))
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
    MISSING_LANGUAGE_MODEL,
    MISSING_VISION_MODEL,
    NETWORK_ERROR,
    INVALID_REPO_FORMAT,
    MULTIPLE_LANGUAGE_MODELS,
    MULTIPLE_VISION_MODELS
}
