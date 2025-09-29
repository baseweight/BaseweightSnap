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
    val modelType: ModelType = ModelType.VISION_ENCODER
)

enum class ModelType {
    VISION_ENCODER, EMBED_TOKENS, DECODER_MODEL, TOKENIZER
}

// Container class for SmolVLM Model set
data class SmolVLMModelSet(
    val name: String,
    val visionEncoder: Model,
    val embedTokens: Model,
    val decoderModel: Model,
    val tokenizer: Model
) {
    val visionEncoderId: String get() = visionEncoder.id
    val embedTokensId: String get() = embedTokens.id
    val decoderModelId: String get() = decoderModel.id
    val tokenizerId: String get() = tokenizer.id
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
        
        // HuggingFace URLs for the ONNX models
        private const val VISION_ENCODER_URL = "https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/resolve/main/onnx/vision_encoder_q4.onnx"
        private const val EMBED_TOKENS_URL = "https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/resolve/main/onnx/embed_tokens_q4.onnx"
        private const val DECODER_MODEL_URL = "https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/resolve/main/onnx/decoder_model_merged_q4.onnx"
        private const val TOKENIZER_URL = "https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/raw/main/tokenizer.json"
    }

    // List of available model sets
    val availableModels = listOf(
        SmolVLMModelSet(
            name = DEFAULT_MODEL_NAME,
            visionEncoder = Model(
                id = "vision_encoder_q4",
                name = "Vision Encoder Q4",
                size = 66_700_000L, // 66.7 MB
                isDefault = true,
                modelType = ModelType.VISION_ENCODER
            ),
            embedTokens = Model(
                id = "embed_tokens_q4",
                name = "Embed Tokens Q4",
                size = 189_000_000L, // 189 MB
                isDefault = true,
                modelType = ModelType.EMBED_TOKENS
            ),
            decoderModel = Model(
                id = "decoder_model_merged_q4",
                name = "Decoder Model Q4",
                size = 229_000_000L, // 229 MB
                isDefault = true,
                modelType = ModelType.DECODER_MODEL
            ),
            tokenizer = Model(
                id = "tokenizer",
                name = "Tokenizer",
                size = 2_400_000L, // ~2.4 MB
                isDefault = true,
                modelType = ModelType.TOKENIZER
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

    // Get SmolVLM model set by name
    fun getModelSet(name: String): SmolVLMModelSet? {
        return availableModels.find { it.name == name }
    }

    // Get model by ID
    fun getModel(modelId: String): Model? {
        return availableModels
            .flatMap { listOf(it.visionEncoder, it.embedTokens, it.decoderModel, it.tokenizer) }
            .find { it.id == modelId }
    }

    // Check if a complete model set is downloaded
    fun isModelSetDownloaded(modelName: String): Boolean {
        val modelSet = getModelSet(modelName) ?: return false
        val visionFile = File(getModelPath(modelSet.visionEncoderId))
        val embedFile = File(getModelPath(modelSet.embedTokensId))
        val decoderFile = File(getModelPath(modelSet.decoderModelId))
        val tokenizerFile = File(getTokenizerPath(modelSet.tokenizerId))

        return visionFile.exists() && visionFile.length() > 0 &&
               embedFile.exists() && embedFile.length() > 0 &&
               decoderFile.exists() && decoderFile.length() > 0 &&
               tokenizerFile.exists() && tokenizerFile.length() > 0
    }

    // Get the local path for a model
    fun getModelPath(modelId: String): String {
        val model = getModel(modelId) ?: throw IllegalArgumentException("Invalid model ID: $modelId")
        val extension = if (model.modelType == ModelType.TOKENIZER) ".json" else ".onnx"
        return "${getModelsDirectory().absolutePath}/${modelId}${extension}"
    }

    // Get the local path for a tokenizer specifically
    fun getTokenizerPath(modelId: String): String {
        return "${getModelsDirectory().absolutePath}/${modelId}.json"
    }

    // Get the local path for a model object
    fun getModelPath(model: Model): String {
        return getModelPath(model.id)
    }

    // Download complete model set (4 files)
    fun downloadModelSet(modelName: String): Flow<DownloadProgress> = flow {
        val modelSet = getModelSet(modelName) ?: throw IllegalArgumentException("Invalid model name: $modelName")

        // Download all models sequentially
        downloadModel(modelSet.visionEncoderId, VISION_ENCODER_URL).collect { progress ->
            emit(progress.copy(message = "Downloading vision encoder..."))
        }

        downloadModel(modelSet.embedTokensId, EMBED_TOKENS_URL).collect { progress ->
            emit(progress.copy(message = "Downloading embed tokens..."))
        }

        downloadModel(modelSet.decoderModelId, DECODER_MODEL_URL).collect { progress ->
            emit(progress.copy(message = "Downloading decoder model..."))
        }

        downloadModel(modelSet.tokenizerId, TOKENIZER_URL).collect { progress ->
            emit(progress.copy(message = "Downloading tokenizer..."))
        }

        // Verify all files are present before marking as COMPLETED
        val visionFile = File(getModelPath(modelSet.visionEncoderId))
        val embedFile = File(getModelPath(modelSet.embedTokensId))
        val decoderFile = File(getModelPath(modelSet.decoderModelId))
        val tokenizerFile = File(getTokenizerPath(modelSet.tokenizerId))

        val visionPresent = visionFile.exists()
        val embedPresent = embedFile.exists()
        val decoderPresent = decoderFile.exists()
        val tokenizerPresent = tokenizerFile.exists()

        val totalSize = modelSet.visionEncoder.size + modelSet.embedTokens.size +
                       modelSet.decoderModel.size + modelSet.tokenizer.size

        if (visionPresent && embedPresent && decoderPresent && tokenizerPresent) {
            emit(DownloadProgress(
                progress = 100,
                bytesDownloaded = totalSize,
                totalBytes = totalSize,
                status = DownloadStatus.COMPLETED,
                message = "All models downloaded successfully"
            ))
            Log.d(TAG, "All models downloaded successfully for: $modelName")
        } else {
            // Clean up any incomplete downloads
            if (!visionPresent) visionFile.delete()
            if (!embedPresent) embedFile.delete()
            if (!decoderPresent) decoderFile.delete()
            if (!tokenizerPresent) tokenizerFile.delete()

            emit(DownloadProgress(
                progress = 100,
                bytesDownloaded = totalSize,
                totalBytes = totalSize,
                status = DownloadStatus.ERROR,
                message = "Failed to verify downloaded files"
            ))
            Log.e(TAG, "Failed to verify downloaded files for: $modelName")
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

            Log.d(TAG, "Downloading from URL: $downloadUrl")
            val downloadConnection = URL(downloadUrl).openConnection() as HttpURLConnection
            downloadConnection.instanceFollowRedirects = true
            downloadConnection.setRequestProperty("User-Agent", "BaseweightSnap/1.0")
            downloadConnection.connect()

            val statusCode = downloadConnection.responseCode
            Log.d(TAG, "HTTP Status: $statusCode, Final URL: ${downloadConnection.url}")

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
            Log.d(TAG, "Content-Length header: ${downloadConnection.contentLength}")
            Log.d(TAG, "Content-Length as long: $totalSize")

            // For tokenizer files and other small files, content-length might not be available
            val effectiveSize = if (totalSize <= 0) {
                Log.w(TAG, "Content-Length not available, using model size estimate: ${model.size}")
                model.size
            } else {
                totalSize
            }
            Log.d(TAG, "Download size: $effectiveSize bytes")

            val inputStream = downloadConnection.inputStream
            val outputStream = FileOutputStream(modelFile)
            val buffer = ByteArray(8192)
            var bytesRead: Int
            var downloadedSize: Long = 0

            emit(DownloadProgress(0, 0, effectiveSize, DownloadStatus.DOWNLOADING))

            while (true) {
                bytesRead = inputStream.read(buffer)
                if (bytesRead == -1) break

                outputStream.write(buffer, 0, bytesRead)
                downloadedSize += bytesRead

                // Calculate progress, but cap at 100% and handle case where actual size exceeds estimate
                val progress = if (effectiveSize > 0) {
                    minOf(100, (downloadedSize * 100 / effectiveSize).toInt())
                } else {
                    50 // Default progress for unknown size
                }
                emit(DownloadProgress(progress, downloadedSize, effectiveSize, DownloadStatus.DOWNLOADING))
            }
            
            outputStream.flush()
            outputStream.close()
            inputStream.close()

            val finalFileSize = modelFile.length()
            // Only verify file size if we had a reliable content-length
            if (totalSize > 0 && finalFileSize != totalSize) {
                Log.e(TAG, "Downloaded file size mismatch: Expected $totalSize, got $finalFileSize")
                modelFile.delete()
                throw Exception("Download failed: File size mismatch")
            } else {
                Log.d(TAG, "Downloaded file size: $finalFileSize bytes")
            }

            emit(DownloadProgress(100, finalFileSize, maxOf(finalFileSize, effectiveSize), DownloadStatus.COMPLETED))
            
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading model: ${e.message}", e)
            emit(DownloadProgress(0, 0, model.size, DownloadStatus.ERROR, e.message ?: "Unknown error"))
            if (modelFile.exists()) {
                modelFile.delete()
            }
        }
    }.flowOn(Dispatchers.IO)
} 