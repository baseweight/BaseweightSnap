package ai.baseweight.baseweightsnap

import android.content.Context
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
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
        
        // HuggingFace URLs for the models
        private const val LANGUAGE_MODEL_URL = "https://huggingface.co/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/resolve/main/SmolVLM2-256M-Video-Instruct-Q8_0.gguf"
        private const val VISION_MODEL_URL = "https://huggingface.co/ggml-org/SmolVLM2-256M-Video-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf"
    }

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
        val model = getModel(modelId) ?: throw IllegalArgumentException("Invalid model ID: $modelId")
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
        downloadModel(mtmdModel.languageId, LANGUAGE_MODEL_URL).collect { progress ->
            // Progress is already emitted by downloadModel
        }
        
        // Download vision model
        downloadModel(mtmdModel.visionId, VISION_MODEL_URL).collect { progress ->
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
} 