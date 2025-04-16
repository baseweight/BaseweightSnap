package ai.baseweight.baseweightsnap

import android.content.Context
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class ModelDownloader(private val context: Context) {
    private val client = OkHttpClient()
    private val baseUrl = context.getString(R.string.api_base_url)
    private val apiKey = context.getString(R.string.api_key)

    data class ModelInfo(
        val id: String,
        val filename: String
    )

    private val models = listOf(
        ModelInfo(
            context.getString(R.string.model_id_decoder),
            "decoder_model_merged.onnx"
        ),
        ModelInfo(
            context.getString(R.string.model_id_embed),
            "embed_tokens.onnx"
        ),
        ModelInfo(
            context.getString(R.string.model_id_vision),
            "vision_encoder.onnx"
        )
    )

    private fun getModelsDir(): File {
        val externalFilesDir = context.getExternalFilesDir(null)
        val modelsDir = File(externalFilesDir, "models")
        if (!modelsDir.exists()) {
            modelsDir.mkdirs()
        }
        Log.d(TAG, "Models directory: ${modelsDir.absolutePath}")
        return modelsDir
    }

    private fun getDownloadUrl(modelId: String): String? {
        val url = "$baseUrl/api/models/$modelId/download"
        Log.d(TAG, "Requesting download URL from: $url")
        
        val request = Request.Builder()
            .url(url)
            .addHeader("Authorization", "Bearer $apiKey")
            .build()

        return try {
            val response = client.newCall(request).execute()
            if (!response.isSuccessful) {
                Log.e(TAG, "Failed to get download URL: ${response.code}")
                return null
            }

            val body = response.body?.string()
            if (body == null) {
                Log.e(TAG, "Empty response body when getting download URL")
                return null
            }

            val json = JSONObject(body)
            val downloadUrl = json.getString("download_url")
            Log.d(TAG, "Got download URL: $downloadUrl")
            downloadUrl
        } catch (e: Exception) {
            Log.e(TAG, "Error getting download URL", e)
            null
        }
    }

    private fun downloadFromUrl(url: String, modelFile: File): Boolean {
        Log.d(TAG, "Downloading from S3 URL: $url")
        
        val request = Request.Builder()
            .url(url)
            .build()

        try {
            val response = client.newCall(request).execute()
            if (!response.isSuccessful) {
                Log.e(TAG, "Failed to download from S3: ${response.code}")
                return false
            }

            val body = response.body
            if (body == null) {
                Log.e(TAG, "Empty response body from S3")
                return false
            }

            Log.d(TAG, "Content length: ${body.contentLength()} bytes")
            Log.d(TAG, "Content type: ${body.contentType()}")

            FileOutputStream(modelFile).use { output ->
                body.byteStream().use { input ->
                    val bytesCopied = input.copyTo(output)
                    Log.d(TAG, "Copied $bytesCopied bytes to ${modelFile.absolutePath}")
                }
            }

            Log.d(TAG, "Successfully downloaded to ${modelFile.absolutePath}")
            Log.d(TAG, "Final file size: ${modelFile.length()} bytes")
            return true
        } catch (e: IOException) {
            Log.e(TAG, "Error downloading from S3", e)
            return false
        }
    }

    suspend fun downloadModels(callback: (Boolean, String?) -> Unit) {
        withContext(Dispatchers.Main) {
            callback(true, null) // Start callback
        }

        withContext(Dispatchers.IO) {
            val modelsDir = getModelsDir()
            var success = true
            var errorMessage: String? = null

            for (model in models) {
                if (!success) break

                val modelFile = File(modelsDir, model.filename)
                Log.d(TAG, "Checking model file: ${modelFile.absolutePath}")
                Log.d(TAG, "File exists: ${modelFile.exists()}")
                if (modelFile.exists()) {
                    Log.d(TAG, "Model ${model.filename} already exists, skipping download")
                    continue
                }

                val downloadUrl = getDownloadUrl(model.id)
                if (downloadUrl == null) {
                    success = false
                    errorMessage = "Failed to get download URL for ${model.filename}"
                    continue
                }

                if (!downloadFromUrl(downloadUrl, modelFile)) {
                    success = false
                    errorMessage = "Failed to download ${model.filename} from S3"
                    continue
                }
            }

            // Log final state of all model files
            Log.d(TAG, "Final model files state:")
            for (model in models) {
                val modelFile = File(modelsDir, model.filename)
                Log.d(TAG, "${model.filename}:")
                Log.d(TAG, "  Path: ${modelFile.absolutePath}")
                Log.d(TAG, "  Exists: ${modelFile.exists()}")
                Log.d(TAG, "  Size: ${modelFile.length()} bytes")
            }

            withContext(Dispatchers.Main) {
                callback(success, errorMessage)
            }
        }
    }

    companion object {
        private const val TAG = "ModelDownloader"
    }

    fun getModelPath(filename: String): String {
        val modelsDir = getModelsDir()
        val modelFile = File(modelsDir, filename)
        return modelFile.absolutePath
    }

    fun copyTokenizerFiles(): Pair<String, String> {
        val modelsDir = getModelsDir()
        
        // Copy vocab.json
        val vocabFile = File(modelsDir, "vocab.json")
        if (!vocabFile.exists()) {
            context.assets.open("vocab.json").use { input ->
                FileOutputStream(vocabFile).use { output ->
                    input.copyTo(output)
                }
            }
            Log.d(TAG, "Copied vocab.json to ${vocabFile.absolutePath}")
        }

        // Copy tokenizer.json
        val tokenizerFile = File(modelsDir, "tokenizer.json")
        if (!tokenizerFile.exists()) {
            context.assets.open("tokenizer.json").use { input ->
                FileOutputStream(tokenizerFile).use { output ->
                    input.copyTo(output)
                }
            }
            Log.d(TAG, "Copied tokenizer.json to ${tokenizerFile.absolutePath}")
        }

        return Pair(vocabFile.absolutePath, tokenizerFile.absolutePath)
    }
} 