package ai.baseweight.baseweightsnap

import ai.baseweight.baseweightsnap.models.*
import android.content.Context
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

data class Model(
    val id: String,
    val name: String,
    val size: Long, // in bytes
    val isDefault: Boolean = false,
    val isLanguage: Boolean = false
)

// Container class for MTMD Model pairs
data class MTMDModel(
    val name: String,
    val language: Model,
    val vision: Model
) {
    val languageId: String get() = language.id
    val visionId: String get() = vision.id
}

data class DownloadProgress(
    val progress: Int, // 0-100
    val bytesDownloaded: Long,
    val totalBytes: Long,
    val status: DownloadStatus,
    val message: String = ""
)

enum class DownloadStatus {
    PENDING, DOWNLOADING, COMPLETED, ERROR
}

class ModelManager(private val context: Context) {

    companion object {
        private const val TAG = "ModelManager"
        private const val MODELS_DIR = "models"
        const val DEFAULT_MODEL_NAME = "SmolVLM2-256M-VidInstruct"
        const val DEFAULT_HF_REPO = "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF"

        // HuggingFace URLs for the models (legacy)
        private const val LANGUAGE_MODEL_URL = "https://huggingface.co/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/resolve/main/SmolVLM2-256M-Video-Instruct-Q8_0.gguf"
        private const val VISION_MODEL_URL = "https://huggingface.co/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf"
    }

    // HuggingFace integration
    private val hfApiClient = HuggingFaceApiClient()
    private val metadataManager = ModelMetadataManager(context)

    // List of available models - simplified to just one model pair
    val availableModels = listOf(
        MTMDModel(
            name = DEFAULT_MODEL_NAME,
            language = Model(
                id = "smolvlm2-256m-language",
                name = "SmolVLM",
                size = 175_000_000L, // Approximate size in bytes
                isDefault = true,
                isLanguage = true
            ),
            vision = Model(
                id = "smolvlm2-256m-vision",
                name = "SmolVLM",
                size = 104_000_000L, // Approximate size in bytes
                isDefault = true,
                isLanguage = false
            )
        )
    )

    init {
        // Create models directory in external storage if it doesn't exist
        getModelsDirectory().mkdirs()

        // Migrate existing legacy SmolVLM2 models to metadata
        migrateLegacyModels()
    }

    /**
     * Migrate legacy SmolVLM2 models to HF metadata system
     */
    private fun migrateLegacyModels() {
        try {
            // Check if legacy SmolVLM2 files exist
            val mtmdModel = getMTMDModel(DEFAULT_MODEL_NAME) ?: return
            val languagePath = getModelPath(mtmdModel.languageId)
            val visionPath = getModelPath(mtmdModel.visionId)

            val languageFile = File(languagePath)
            val visionFile = File(visionPath)

            if (languageFile.exists() && visionFile.exists()) {
                // Check if already migrated
                val existing = metadataManager.getAllModels().find {
                    it.hfRepo == DEFAULT_HF_REPO
                }

                if (existing == null) {
                    // Create metadata for legacy model
                    val metadata = HFModelMetadata(
                        id = "legacy-smolvlm2-256m",
                        name = DEFAULT_MODEL_NAME,
                        hfRepo = DEFAULT_HF_REPO,
                        languageFile = "SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
                        visionFile = "mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
                        configFile = null, // config.json not needed for GGUF models
                        downloadDate = languageFile.lastModified(),
                        languageSize = languageFile.length(),
                        visionSize = visionFile.length()
                    )

                    metadataManager.saveMetadata(metadata)

                    // Set as default if no other default exists
                    if (metadataManager.getDefaultModel() == null) {
                        metadataManager.setDefaultModel(metadata.id)
                        Log.d(TAG, "Migrated legacy SmolVLM2 model and set as default")
                    } else {
                        Log.d(TAG, "Migrated legacy SmolVLM2 model")
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error migrating legacy models: ${e.message}", e)
        }
    }

    private fun getModelsDirectory(): File {
        return File(context.getExternalFilesDir(null), MODELS_DIR)
    }

    // Get MTMD model pair by name
    fun getMTMDModel(name: String): MTMDModel? {
        return availableModels.find { it.name == name }
    }

    // Get model by ID
    fun getModel(modelId: String): Model? {
        return availableModels
            .flatMap { listOf(it.language, it.vision) }
            .find { it.id == modelId }
    }

    // Check if a model pair is downloaded
    fun isModelPairDownloaded(modelName: String): Boolean {
        val model = getMTMDModel(modelName) ?: return false
        val languageFile = File(getModelPath(model.languageId))
        val visionFile = File(getModelPath(model.visionId))
        return languageFile.exists() && languageFile.length() > 0 &&
               visionFile.exists() && visionFile.length() > 0
    }

    // Get the local path for a model
    fun getModelPath(modelId: String): String {
        getModel(modelId) ?: throw IllegalArgumentException("Invalid model ID: $modelId")
        return "${getModelsDirectory().absolutePath}/${modelId}.gguf"
    }

    // Get the local path for a model object
    fun getModelPath(model: Model): String {
        return getModelPath(model.id)
    }

    // Download both language and vision models for a model pair
    fun downloadModelPair(modelName: String): Flow<DownloadProgress> = flow {
        val mtmdModel = getMTMDModel(modelName) ?: throw IllegalArgumentException("Invalid model name: $modelName")
        
        // Download language model first
        downloadModel(mtmdModel.languageId, LANGUAGE_MODEL_URL).collect { _ ->
            // Progress is already emitted by downloadModel
        }

        // Download vision model
        downloadModel(mtmdModel.visionId, VISION_MODEL_URL).collect { _ ->
            // Progress is already emitted by downloadModel
        }
        
        // Verify both files are present before marking as COMPLETED
        val languageFile = File(getModelPath(mtmdModel.languageId))
        val visionFile = File(getModelPath(mtmdModel.visionId))
        
        val languagePresent = languageFile.exists()
        val visionPresent = visionFile.exists()
        
        val totalSize = mtmdModel.language.size + mtmdModel.vision.size
        if (languagePresent && visionPresent) {
            emit(DownloadProgress(
                progress = 100,
                bytesDownloaded = totalSize,
                totalBytes = totalSize,
                status = DownloadStatus.COMPLETED,
                message = "Both models downloaded successfully"
            ))
            Log.d(TAG, "Both language and vision models downloaded successfully for: $modelName")
        } else {
            // If either file is missing, clean up and mark as error
            if (!languagePresent) languageFile.delete()
            if (!visionPresent) visionFile.delete()
            
            emit(DownloadProgress(
                progress = 100,
                bytesDownloaded = totalSize,
                totalBytes = totalSize,
                status = DownloadStatus.ERROR,
                message = "Failed to verify downloaded files"
            ))
            Log.e(TAG, "Failed to verify downloaded files for: $modelName. Language: $languagePresent, Vision: $visionPresent")
        }
    }.flowOn(Dispatchers.IO)

    // Download a single model
    private fun downloadModel(modelId: String, downloadUrl: String): Flow<DownloadProgress> = flow {
        val model = getModel(modelId) ?: throw IllegalArgumentException("Invalid model ID")
        val modelFile = File(getModelPath(model))
        
        // Create parent directories if they don't exist
        modelFile.parentFile?.mkdirs()

        try {
            emit(DownloadProgress(0, 0, model.size, DownloadStatus.PENDING, "Connecting to download server..."))
            
            val downloadConnection = URL(downloadUrl).openConnection() as HttpURLConnection
            downloadConnection.connect()
            
            val statusCode = downloadConnection.responseCode
            if (statusCode != HttpURLConnection.HTTP_OK) {
                val errorStream = downloadConnection.errorStream
                val errorMessage = errorStream?.bufferedReader()?.use { it.readText() } ?: "No error message available"
                val errorDetails = "HTTP ${statusCode} - ${downloadConnection.responseMessage}\n${errorMessage}"
                Log.e(TAG, "Download Error: $errorDetails")
                
                emit(DownloadProgress(
                    progress = 0,
                    bytesDownloaded = 0,
                    totalBytes = model.size,
                    status = DownloadStatus.ERROR,
                    message = "Download failed: $errorDetails"
                ))
                
                throw Exception("Download failed with HTTP ${statusCode}: ${downloadConnection.responseMessage}")
            }
            
            val totalSize = downloadConnection.contentLength.toLong()
            if (totalSize <= 0) {
                Log.e(TAG, "Invalid content length received: $totalSize")
                throw Exception("Invalid content length received from server")
            }
            Log.d(TAG, "Download size: $totalSize bytes")
            
            val inputStream = downloadConnection.inputStream
            val outputStream = FileOutputStream(modelFile)
            val buffer = ByteArray(8192)
            var bytesRead: Int
            var downloadedSize: Long = 0
            
            emit(DownloadProgress(0, 0, totalSize, DownloadStatus.DOWNLOADING))
            
            while (true) {
                bytesRead = inputStream.read(buffer)
                if (bytesRead == -1) break
                
                outputStream.write(buffer, 0, bytesRead)
                downloadedSize += bytesRead
                
                val progress = (downloadedSize * 100 / totalSize).toInt()
                emit(DownloadProgress(progress, downloadedSize, totalSize, DownloadStatus.DOWNLOADING))
            }
            
            outputStream.flush()
            outputStream.close()
            inputStream.close()
            
            val finalFileSize = modelFile.length()
            if (finalFileSize != totalSize) {
                Log.e(TAG, "Downloaded file size mismatch: Expected $totalSize, got $finalFileSize")
                modelFile.delete()
                throw Exception("Download failed: File size mismatch")
            }
            
            emit(DownloadProgress(100, totalSize, totalSize, DownloadStatus.COMPLETED))
            
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading model: ${e.message}", e)
            emit(DownloadProgress(0, 0, model.size, DownloadStatus.ERROR, e.message ?: "Unknown error"))
            if (modelFile.exists()) {
                modelFile.delete()
            }
        }
    }.flowOn(Dispatchers.IO)

    // ========== NEW: HuggingFace Model Management ==========

    /**
     * Validate a HuggingFace repository for VLM compatibility
     */
    suspend fun validateHFRepo(repo: String): ValidationResult = withContext(Dispatchers.IO) {
        Log.d(TAG, "Validating HF repository: $repo")
        hfApiClient.validateModel(repo)
    }

    /**
     * Download a model from HuggingFace repository
     * Returns the metadata of the downloaded model
     */
    suspend fun downloadFromHuggingFace(repo: String): Flow<DownloadProgress> = flow {
        Log.d(TAG, "Starting download from HF repo: $repo")

        // Step 1: Validate repository
        emit(DownloadProgress(0, 0, 0, DownloadStatus.PENDING, "Validating repository..."))
        val validation = hfApiClient.validateModel(repo)

        if (validation is ValidationResult.Invalid) {
            val errorMsg = when (validation.error) {
                ValidationError.REPO_NOT_FOUND -> "Repository not found"
                ValidationError.MISSING_LANGUAGE_MODEL -> "Missing language model (GGUF)"
                ValidationError.MISSING_VISION_MODEL -> "Missing vision model (mmproj GGUF)"
                ValidationError.INVALID_REPO_FORMAT -> "Invalid repository format. Use: orgName/repository"
                else -> "Validation failed"
            }
            emit(DownloadProgress(0, 0, 0, DownloadStatus.ERROR, errorMsg))
            return@flow
        }

        val files = (validation as ValidationResult.Valid).files
        val totalSize = files.totalSize

        emit(DownloadProgress(1, 0, totalSize, DownloadStatus.PENDING, "Downloading files..."))

        // Step 2: Download language model
        val languageFileName = files.languageFile.path.substringAfterLast("/")
        val languagePath = "${getModelsDirectory()}/${repo.replace("/", "_")}_$languageFileName"
        val languageUrl = hfApiClient.getDownloadUrl(repo, files.languageFile.path)
        var lastLanguageProgress = -1
        downloadFile(languageUrl, languagePath, files.languageFile.size).collect { progress ->
            // Skip COMPLETED status from individual file - we'll emit overall COMPLETED at the end
            if (progress.status == DownloadStatus.COMPLETED) return@collect

            val adjustedProgress = (progress.progress * 0.5).toInt() + 1 // 1-51%
            // Only emit if adjusted progress changed
            if (adjustedProgress != lastLanguageProgress) {
                emit(progress.copy(
                    progress = adjustedProgress,
                    message = "Downloading language model...",
                    status = DownloadStatus.DOWNLOADING
                ))
                lastLanguageProgress = adjustedProgress
            }
        }

        // Step 3: Download vision model
        val visionFileName = files.visionFile.path.substringAfterLast("/")
        val visionPath = "${getModelsDirectory()}/${repo.replace("/", "_")}_$visionFileName"
        val visionUrl = hfApiClient.getDownloadUrl(repo, files.visionFile.path)
        var lastVisionProgress = -1
        downloadFile(visionUrl, visionPath, files.visionFile.size).collect { progress ->
            // Skip COMPLETED status from individual file - we'll emit overall COMPLETED at the end
            if (progress.status == DownloadStatus.COMPLETED) return@collect

            val adjustedProgress = (progress.progress * 0.5).toInt() + 51 // 51-100%
            // Only emit if adjusted progress changed
            if (adjustedProgress != lastVisionProgress) {
                emit(progress.copy(
                    progress = adjustedProgress,
                    message = "Downloading vision model...",
                    status = DownloadStatus.DOWNLOADING
                ))
                lastVisionProgress = adjustedProgress
            }
        }

        // Step 4: Save metadata
        emit(DownloadProgress(97, totalSize, totalSize, DownloadStatus.DOWNLOADING, "Saving metadata..."))

        val repoName = repo.substringAfterLast("/")
        val metadata = HFModelMetadata(
            name = repoName,
            hfRepo = repo,
            languageFile = languageFileName,
            visionFile = visionFileName,
            configFile = null, // config.json not needed for GGUF models
            languageSize = files.languageFile.size,
            visionSize = files.visionFile.size
        )

        metadataManager.saveMetadata(metadata)
        Log.d(TAG, "Saved metadata for model: ${metadata.id}")

        // Step 5: Set as default if no other models exist
        if (metadataManager.getAllModels().size == 1) {
            metadataManager.setDefaultModel(metadata.id)
            Log.d(TAG, "Set as default model: ${metadata.id}")
        }

        emit(DownloadProgress(100, totalSize, totalSize, DownloadStatus.COMPLETED, "Download complete!"))
    }.flowOn(Dispatchers.IO)

    /**
     * Download a single file with progress tracking
     */
    private fun downloadFile(url: String, destPath: String, expectedSize: Long): Flow<DownloadProgress> = flow {
        val file = File(destPath)
        file.parentFile?.mkdirs()

        try {
            val connection = URL(url).openConnection() as HttpURLConnection
            connection.connect()

            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                throw Exception("HTTP ${connection.responseCode}: ${connection.responseMessage}")
            }

            val totalSize = connection.contentLength.toLong()
            val inputStream = connection.inputStream
            val outputStream = FileOutputStream(file)
            val buffer = ByteArray(8192)
            var bytesRead: Int
            var downloadedSize: Long = 0
            var lastProgress = 0

            emit(DownloadProgress(0, 0, totalSize, DownloadStatus.DOWNLOADING))

            while (true) {
                bytesRead = inputStream.read(buffer)
                if (bytesRead == -1) break

                outputStream.write(buffer, 0, bytesRead)
                downloadedSize += bytesRead

                val progress = if (totalSize > 0) {
                    (downloadedSize * 100 / totalSize).toInt()
                } else {
                    0
                }

                // Only emit if progress percentage changed
                if (progress != lastProgress) {
                    emit(DownloadProgress(progress, downloadedSize, totalSize, DownloadStatus.DOWNLOADING))
                    lastProgress = progress
                }
            }

            outputStream.flush()
            outputStream.close()
            inputStream.close()

            emit(DownloadProgress(100, totalSize, totalSize, DownloadStatus.COMPLETED))

        } catch (e: Exception) {
            Log.e(TAG, "Error downloading file: ${e.message}", e)
            if (file.exists()) file.delete()
            emit(DownloadProgress(0, 0, expectedSize, DownloadStatus.ERROR, e.message ?: "Download failed"))
        }
    }.flowOn(Dispatchers.IO)

    /**
     * Get all downloaded models (from metadata)
     */
    fun listDownloadedModels(): List<HFModelMetadata> {
        return metadataManager.getAllModels()
    }

    /**
     * Get the default model
     */
    fun getDefaultModel(): HFModelMetadata? {
        return metadataManager.getDefaultModel()
    }

    /**
     * Set a model as default
     */
    fun setDefaultModel(modelId: String) {
        metadataManager.setDefaultModel(modelId)
    }

    /**
     * Delete a model and its files
     */
    suspend fun deleteModel(modelId: String): Result<Unit> = withContext(Dispatchers.IO) {
        return@withContext try {
            val metadata = metadataManager.getModelById(modelId)
                ?: return@withContext Result.failure(Exception("Model not found: $modelId"))

            // Delete files
            val prefix = metadata.hfRepo.replace("/", "_")
            val modelsDir = getModelsDirectory()

            val languageFile = File(modelsDir, "${prefix}_${metadata.languageFile}")
            val visionFile = File(modelsDir, "${prefix}_${metadata.visionFile}")

            languageFile.delete()
            visionFile.delete()

            // Delete metadata
            metadataManager.deleteModel(modelId)

            Log.d(TAG, "Deleted model: $modelId")
            Result.success(Unit)
        } catch (e: Exception) {
            Log.e(TAG, "Error deleting model: ${e.message}", e)
            Result.failure(e)
        }
    }

    /**
     * Get file paths for a HF model
     * Returns Pair(languagePath, visionPath)
     */
    fun getHFModelPaths(modelId: String): Pair<String, String>? {
        val metadata = metadataManager.getModelById(modelId) ?: return null
        val modelsDir = getModelsDirectory()

        // Special handling for legacy migrated model
        if (modelId == "legacy-smolvlm2-256m") {
            val mtmdModel = getMTMDModel(DEFAULT_MODEL_NAME)!!
            return Pair(
                getModelPath(mtmdModel.languageId),
                getModelPath(mtmdModel.visionId)
            )
        }

        // Standard HF model path
        val prefix = metadata.hfRepo.replace("/", "_")
        val languagePath = File(modelsDir, "${prefix}_${metadata.languageFile}").absolutePath
        val visionPath = File(modelsDir, "${prefix}_${metadata.visionFile}").absolutePath

        return Pair(languagePath, visionPath)
    }

    /**
     * Check if any HF models are downloaded
     */
    fun hasDownloadedModels(): Boolean {
        return metadataManager.hasModels()
    }
}