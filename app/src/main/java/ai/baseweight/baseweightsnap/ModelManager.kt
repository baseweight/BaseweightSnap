package ai.baseweight.baseweightsnap

import android.content.Context
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.onEach
import org.json.JSONObject
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
        private const val BASE_URL = "https://api.baseweight.ai/api"
        private const val MODELS_DIR = "models"
        const val DEFAULT_MODEL_NAME = "SmolVLM-2.2B-Instruct"
    }

    private val apiKey: String by lazy {
        context.getString(R.string.baseweight_api_key)
    }

    // List of available models - we'll hardcode IDs and names but fetch actual details from the API
    val availableModels = listOf(
        MTMDModel(
            name = DEFAULT_MODEL_NAME,
            language = Model(
                id = context.getString(R.string.smolvlm2_2_2b_language),
                name = "SmolVLM",
                size = 1_838_620_000L, // 1838.62 MB
                isDefault = true
            ),
            vision = Model(
                id = context.getString(R.string.smolvlm2_2_2b_vision),
                name = "SmolVLM",
                size = 565_070_000L, // 565.07 MB
                isDefault = true
            )
        )
    )

    init {
        // Create models directory in external storage if it doesn't exist
        getModelsDirectory().mkdirs()
    }

    private fun getModelsDirectory(): File {
        // Use external storage directory
        return File(context.getExternalFilesDir(null), MODELS_DIR)
    }

    // Get MTMD model pair by name
    fun getMTMDModel(name: String): MTMDModel? {
        return availableModels.find { it.name == name }
    }

    // Get model by ID
    fun getModel(modelId: String): Model? {
        return availableModels
            .flatMap { listOf(it.language.copy(isLanguage = true), it.vision.copy(isLanguage = false)) }
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
        downloadModel(mtmdModel.languageId).collect { progress ->
            // Progress is already emitted by downloadModel
        }
        
        // Download vision model
        downloadModel(mtmdModel.visionId).collect { progress ->
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

    // Download a single model (helper for backward compatibility)
    private fun downloadModel(modelId: String): Flow<DownloadProgress> = flow {
        val model = getModel(modelId) ?: throw IllegalArgumentException("Invalid model ID")
        val modelFile = File(getModelPath(model))
        
        // Create parent directories if they don't exist
        modelFile.parentFile?.mkdirs()

        try {
            emit(DownloadProgress(0, 0, model.size, DownloadStatus.PENDING, "Requesting download URL..."))
            
            // Step 1: Get the pre-signed URL from the API
            val apiUrl = "$BASE_URL/models/$modelId/download"
            Log.d(TAG, "Requesting download URL from: $apiUrl")
            
            val apiConnection = URL(apiUrl).openConnection() as HttpURLConnection
            apiConnection.setRequestProperty("Authorization", "Bearer $apiKey")
            apiConnection.connect()
            
            // Check if we got a successful response from the API
            if (apiConnection.responseCode != HttpURLConnection.HTTP_OK) {
                val errorMessage = apiConnection.errorStream?.bufferedReader()?.use { it.readText() } ?: "Unknown error"
                Log.e(TAG, "API Error: ${apiConnection.responseCode} - $errorMessage")
                throw Exception("API Error: ${apiConnection.responseCode} - $errorMessage")
            }
            
            // Parse the response to get the pre-signed URL
            val responseText = apiConnection.inputStream.bufferedReader().use { it.readText() }
            Log.d(TAG, "API Response: $responseText")
            
            val jsonResponse = JSONObject(responseText)
            
            // Extract the pre-signed URL matching the Python script's field name
            val preSignedUrl = jsonResponse.getString("download_url")
            Log.d(TAG, "Got pre-signed download URL: $preSignedUrl")
            
            emit(DownloadProgress(0, 0, model.size, DownloadStatus.PENDING, "Connecting to download server..."))
            
            // Step 2: Download the file using the pre-signed URL
            val downloadConnection = URL(preSignedUrl).openConnection() as HttpURLConnection
            downloadConnection.connect()
            
            // Check HTTP status code and get detailed error information
            val statusCode = downloadConnection.responseCode
            if (statusCode != HttpURLConnection.HTTP_OK) {
                val errorStream = downloadConnection.errorStream
                val errorMessage = errorStream?.bufferedReader()?.use { it.readText() } ?: "No error message available"
                val errorDetails = "HTTP ${statusCode} - ${downloadConnection.responseMessage}\n${errorMessage}"
                Log.e(TAG, "Download Error: $errorDetails")
                
                // Emit error progress before throwing exception
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
            Log.d(TAG, "Download size from S3: $totalSize bytes")
            
            val inputStream = downloadConnection.inputStream
            val outputStream = FileOutputStream(modelFile)
            val buffer = ByteArray(8192)
            var bytesRead: Int
            var downloadedSize: Long = 0
            
            emit(DownloadProgress(0, 0, totalSize, DownloadStatus.DOWNLOADING))
            
            // Download the file in chunks
            while (true) {
                bytesRead = inputStream.read(buffer)
                if (bytesRead == -1) break
                
                outputStream.write(buffer, 0, bytesRead)
                downloadedSize += bytesRead
                
                val progress = (downloadedSize * 100 / totalSize).toInt()
                emit(DownloadProgress(progress, downloadedSize, totalSize, DownloadStatus.DOWNLOADING))
            }
            
            // Ensure the file is completely written before closing
            outputStream.flush()
            outputStream.close()
            inputStream.close()
            
            // Verify the file size matches what we expected
            val finalFileSize = modelFile.length()
            if (finalFileSize != totalSize) {
                Log.e(TAG, "Downloaded file size mismatch: Expected $totalSize, got $finalFileSize")
                modelFile.delete() // Delete the corrupted file
                throw Exception("Download failed: File size mismatch")
            }
            
            emit(DownloadProgress(100, totalSize, totalSize, DownloadStatus.COMPLETED))
            
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading model: ${e.message}", e)
            emit(DownloadProgress(0, 0, model.size, DownloadStatus.ERROR, e.message ?: "Unknown error"))
            if (modelFile.exists()) {
                modelFile.delete() // Delete partial file
            }
        }
    }.flowOn(Dispatchers.IO)
} 