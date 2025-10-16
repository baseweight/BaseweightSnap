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

data class ModelFile(
    val filename: String,
    val size: Long // approximate size in bytes
)

data class Model(
    val id: String,
    val name: String,
    val repoUrl: String,
    val files: List<ModelFile>,
    val isDefault: Boolean = false
) {
    val totalSize: Long get() = files.sumOf { it.size }
}

data class DownloadProgress(
    val progress: Int, // 0-100
    val bytesDownloaded: Long,
    val totalBytes: Long,
    val status: DownloadStatus,
    val currentFile: String = "",
    val message: String = ""
)

enum class DownloadStatus {
    PENDING, DOWNLOADING, COMPLETED, ERROR
}

class ModelManager(private val context: Context) {

    companion object {
        private const val TAG = "ModelManager"
        private const val MODELS_DIR = "models"
        const val DEFAULT_MODEL_NAME = "nanoVLM-230M-8k-executorch"

        // HuggingFace repo for nanoVLM ExecuTorch models (XNNPACK quantized)
        private const val NANOVLM_REPO = "https://huggingface.co/infil00p/nanoVLM-230M-8k-executorch/resolve/main/xnnpack"

        // HuggingFace repo for tokenizer
        private const val TOKENIZER_REPO = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main"
    }

    // List of available models
    val availableModels = listOf(
        Model(
            id = "nanovlm-230m-8k",
            name = DEFAULT_MODEL_NAME,
            repoUrl = NANOVLM_REPO,
            files = listOf(
                ModelFile("vision_encoder.pte", 88_000_000L),
                ModelFile("modality_projector.pte", 6_800_000L),
                ModelFile("token_embedding.pte", 109_000_000L),
                ModelFile("language_decoder_prefill.pte", 103_000_000L),
                ModelFile("language_decoder_decode.pte", 103_000_000L),
                ModelFile("lm_head.pte", 109_000_000L),
                ModelFile("config.json", 3_200L)
            ),
            isDefault = true
        )
    )

    // Tokenizer files (downloaded separately from different repo)
    private val tokenizerFiles = listOf(
        ModelFile("tokenizer.json", 800_000L),
        ModelFile("tokenizer_config.json", 20_000L)
    )

    init {
        // Create models directory in external storage if it doesn't exist
        getModelsDirectory().mkdirs()
        getTokenizerDirectory().mkdirs()
    }

    private fun getModelsDirectory(): File {
        return File(context.getExternalFilesDir(null), MODELS_DIR)
    }

    private fun getTokenizerDirectory(): File {
        return File(context.getExternalFilesDir(null), "tokenizer")
    }

    // Get model by name
    fun getModel(name: String): Model? {
        return availableModels.find { it.name == name }
    }

    // Check if a model is fully downloaded
    fun isModelDownloaded(modelName: String): Boolean {
        val model = getModel(modelName) ?: return false
        val modelsDir = getModelsDirectory()
        val tokenizerDir = getTokenizerDirectory()

        // Check all model files exist and have non-zero size
        val allModelFilesPresent = model.files.all { modelFile ->
            val file = File(modelsDir, modelFile.filename)
            file.exists() && file.length() > 0
        }

        // Check tokenizer files exist
        val allTokenizerFilesPresent = tokenizerFiles.all { tokenizerFile ->
            val file = File(tokenizerDir, tokenizerFile.filename)
            file.exists() && file.length() > 0
        }

        return allModelFilesPresent && allTokenizerFilesPresent
    }

    // Get the local path for the models directory
    fun getModelDirectoryPath(): String {
        return getModelsDirectory().absolutePath
    }

    // Get the local path for the tokenizer directory
    fun getTokenizerPath(): String {
        return File(getTokenizerDirectory(), "tokenizer.json").absolutePath
    }

    // Download model with all required files
    fun downloadModel(modelName: String): Flow<DownloadProgress> = flow {
        val model = getModel(modelName) ?: throw IllegalArgumentException("Invalid model name: $modelName")

        val modelsDir = getModelsDirectory()
        val tokenizerDir = getTokenizerDirectory()

        // Calculate total size (model files + tokenizer files)
        val totalSize = model.totalSize + tokenizerFiles.sumOf { it.size }
        var totalDownloaded = 0L

        emit(DownloadProgress(
            progress = 0,
            bytesDownloaded = 0,
            totalBytes = totalSize,
            status = DownloadStatus.PENDING,
            message = "Preparing to download ${model.files.size + tokenizerFiles.size} files..."
        ))

        try {
            // Download all model files
            for (modelFile in model.files) {
                val fileUrl = "${model.repoUrl}/${modelFile.filename}"
                val localFile = File(modelsDir, modelFile.filename)

                Log.d(TAG, "Downloading ${modelFile.filename} from $fileUrl")

                downloadFile(
                    url = fileUrl,
                    destinationFile = localFile,
                    expectedSize = modelFile.size
                ).collect { fileProgress ->
                    val overallProgress = ((totalDownloaded + fileProgress.bytesDownloaded) * 100 / totalSize).toInt()
                    emit(DownloadProgress(
                        progress = overallProgress,
                        bytesDownloaded = totalDownloaded + fileProgress.bytesDownloaded,
                        totalBytes = totalSize,
                        status = DownloadStatus.DOWNLOADING,
                        currentFile = modelFile.filename,
                        message = "Downloading ${modelFile.filename}..."
                    ))

                    // If this file completed, add to total
                    if (fileProgress.status == DownloadStatus.COMPLETED) {
                        totalDownloaded += modelFile.size
                    }
                }
            }

            // Download tokenizer files
            for (tokenizerFile in tokenizerFiles) {
                val fileUrl = "${TOKENIZER_REPO}/${tokenizerFile.filename}"
                val localFile = File(tokenizerDir, tokenizerFile.filename)

                Log.d(TAG, "Downloading ${tokenizerFile.filename} from $fileUrl")

                downloadFile(
                    url = fileUrl,
                    destinationFile = localFile,
                    expectedSize = tokenizerFile.size
                ).collect { fileProgress ->
                    val overallProgress = ((totalDownloaded + fileProgress.bytesDownloaded) * 100 / totalSize).toInt()
                    emit(DownloadProgress(
                        progress = overallProgress,
                        bytesDownloaded = totalDownloaded + fileProgress.bytesDownloaded,
                        totalBytes = totalSize,
                        status = DownloadStatus.DOWNLOADING,
                        currentFile = tokenizerFile.filename,
                        message = "Downloading ${tokenizerFile.filename}..."
                    ))

                    // If this file completed, add to total
                    if (fileProgress.status == DownloadStatus.COMPLETED) {
                        totalDownloaded += tokenizerFile.size
                    }
                }
            }

            // Verify ALL files are present before marking as COMPLETED
            val allFilesPresent = model.files.all { modelFile ->
                val file = File(modelsDir, modelFile.filename)
                file.exists() && file.length() > 0
            } && tokenizerFiles.all { tokenizerFile ->
                val file = File(tokenizerDir, tokenizerFile.filename)
                file.exists() && file.length() > 0
            }

            if (allFilesPresent) {
                emit(DownloadProgress(
                    progress = 100,
                    bytesDownloaded = totalSize,
                    totalBytes = totalSize,
                    status = DownloadStatus.COMPLETED,
                    message = "All files downloaded successfully"
                ))
                Log.d(TAG, "All model and tokenizer files downloaded successfully for: $modelName")
            } else {
                // If any file is missing, clean up and mark as error
                val missingFiles = mutableListOf<String>()
                model.files.forEach { modelFile ->
                    val file = File(modelsDir, modelFile.filename)
                    if (!file.exists() || file.length() == 0L) {
                        missingFiles.add(modelFile.filename)
                        file.delete()
                    }
                }
                tokenizerFiles.forEach { tokenizerFile ->
                    val file = File(tokenizerDir, tokenizerFile.filename)
                    if (!file.exists() || file.length() == 0L) {
                        missingFiles.add(tokenizerFile.filename)
                        file.delete()
                    }
                }

                emit(DownloadProgress(
                    progress = 100,
                    bytesDownloaded = totalSize,
                    totalBytes = totalSize,
                    status = DownloadStatus.ERROR,
                    message = "Failed to verify files: ${missingFiles.joinToString(", ")}"
                ))
                Log.e(TAG, "Missing files after download: ${missingFiles.joinToString(", ")}")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error downloading model: ${e.message}", e)
            emit(DownloadProgress(
                progress = 0,
                bytesDownloaded = 0,
                totalBytes = totalSize,
                status = DownloadStatus.ERROR,
                message = e.message ?: "Unknown error"
            ))
        }
    }.flowOn(Dispatchers.IO)

    // Download a single file
    private fun downloadFile(
        url: String,
        destinationFile: File,
        expectedSize: Long
    ): Flow<DownloadProgress> = flow {
        // Create parent directories if they don't exist
        destinationFile.parentFile?.mkdirs()

        try {
            emit(DownloadProgress(0, 0, expectedSize, DownloadStatus.PENDING, message = "Connecting..."))

            val downloadConnection = URL(url).openConnection() as HttpURLConnection
            downloadConnection.connect()

            val statusCode = downloadConnection.responseCode
            if (statusCode != HttpURLConnection.HTTP_OK) {
                val errorStream = downloadConnection.errorStream
                val errorMessage = errorStream?.bufferedReader()?.use { it.readText() } ?: "No error message available"
                val errorDetails = "HTTP ${statusCode} - ${downloadConnection.responseMessage}\n${errorMessage}"
                Log.e(TAG, "Download Error for ${destinationFile.name}: $errorDetails")

                emit(DownloadProgress(
                    progress = 0,
                    bytesDownloaded = 0,
                    totalBytes = expectedSize,
                    status = DownloadStatus.ERROR,
                    message = "Download failed: $errorDetails"
                ))

                throw Exception("Download failed with HTTP ${statusCode}: ${downloadConnection.responseMessage}")
            }

            val totalSize = downloadConnection.contentLength.toLong()

            // For small files (like JSON), HuggingFace may return -1 for contentLength
            // In this case, use expectedSize as estimate and don't fail
            val effectiveSize = if (totalSize <= 0) {
                Log.w(TAG, "Content length not provided for ${destinationFile.name}, using expected size: $expectedSize")
                expectedSize
            } else {
                Log.d(TAG, "Download size for ${destinationFile.name}: $totalSize bytes")
                totalSize
            }

            val inputStream = downloadConnection.inputStream
            val outputStream = FileOutputStream(destinationFile)
            val buffer = ByteArray(8192)
            var bytesRead: Int
            var downloadedSize: Long = 0

            emit(DownloadProgress(0, 0, effectiveSize, DownloadStatus.DOWNLOADING))

            while (true) {
                bytesRead = inputStream.read(buffer)
                if (bytesRead == -1) break

                outputStream.write(buffer, 0, bytesRead)
                downloadedSize += bytesRead

                val progress = (downloadedSize * 100 / effectiveSize).toInt().coerceIn(0, 100)
                emit(DownloadProgress(progress, downloadedSize, effectiveSize, DownloadStatus.DOWNLOADING))
            }

            outputStream.flush()
            outputStream.close()
            inputStream.close()

            val finalFileSize = destinationFile.length()

            // Only validate file size if we got a valid contentLength from server
            if (totalSize > 0 && finalFileSize != totalSize) {
                Log.e(TAG, "Downloaded file size mismatch for ${destinationFile.name}: Expected $totalSize, got $finalFileSize")
                destinationFile.delete()
                throw Exception("Download failed: File size mismatch")
            }

            Log.d(TAG, "Successfully downloaded ${destinationFile.name}: $finalFileSize bytes")
            emit(DownloadProgress(100, finalFileSize, finalFileSize, DownloadStatus.COMPLETED))

        } catch (e: Exception) {
            Log.e(TAG, "Error downloading file ${destinationFile.name}: ${e.message}", e)
            emit(DownloadProgress(0, 0, expectedSize, DownloadStatus.ERROR, message = e.message ?: "Unknown error"))
            if (destinationFile.exists()) {
                destinationFile.delete()
            }
        }
    }.flowOn(Dispatchers.IO)
}
