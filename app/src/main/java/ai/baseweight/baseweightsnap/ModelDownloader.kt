package ai.baseweight.baseweightsnap

import android.content.Context
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
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

                try {
                    val url = "$baseUrl/api/models/${model.id}"
                    Log.d(TAG, "Attempting to download from URL: $url")
                    
                    val request = Request.Builder()
                        .url(url)
                        .addHeader("Authorization", "Bearer $apiKey")
                        .build()

                    Log.d(TAG, "Making request for ${model.filename}")
                    val response = client.newCall(request).execute()
                    Log.d(TAG, "Response code: ${response.code}")
                    Log.d(TAG, "Response message: ${response.message}")
                    
                    if (!response.isSuccessful) {
                        success = false
                        errorMessage = "Failed to download ${model.filename}: ${response.code}"
                        Log.e(TAG, "Download failed: $errorMessage")
                        continue
                    }

                    val body = response.body
                    if (body == null) {
                        success = false
                        errorMessage = "Empty response body for ${model.filename}"
                        Log.e(TAG, "Empty response body")
                        continue
                    }

                    Log.d(TAG, "Content length: ${body.contentLength()} bytes")
                    Log.d(TAG, "Content type: ${body.contentType()}")

                    FileOutputStream(modelFile).use { output ->
                        body.byteStream().use { input ->
                            val bytesCopied = input.copyTo(output)
                            Log.d(TAG, "Copied $bytesCopied bytes to ${modelFile.absolutePath}")
                        }
                    }

                    Log.d(TAG, "Successfully downloaded ${model.filename} to ${modelFile.absolutePath}")
                    Log.d(TAG, "Final file size: ${modelFile.length()} bytes")
                } catch (e: IOException) {
                    success = false
                    errorMessage = "Error downloading ${model.filename}: ${e.message}"
                    Log.e(TAG, "Download error", e)
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
} 