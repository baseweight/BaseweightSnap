package ai.baseweight.baseweightsnap

import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.channels.onFailure
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class NanoVLM_Android {
    private val tag: String? = this::class.simpleName

    private val runLoop: CoroutineDispatcher = Executors.newSingleThreadExecutor {
        thread(start = false, name = "NanoVLM-RunLoop") {
            Log.d(tag, "Dedicated thread for NanoVLM native code: ${Thread.currentThread().name}")

            // No-op if called more than once.
            System.loadLibrary("baseweightsnap")

            Log.d(tag, "NanoVLM native library loaded")

            it.run()
        }.apply {
            uncaughtExceptionHandler = Thread.UncaughtExceptionHandler { _, exception: Throwable ->
                Log.e(tag, "Unhandled exception in NanoVLM thread", exception)
            }
        }
    }.asCoroutineDispatcher()

    // JNI native methods matching nanovlm_android.cpp exports
    private external fun nativeLoadModels(modelDirPath: String, tokenizerPath: String): Boolean
    private external fun nativeProcessImageFromBuffer(buffer: ByteBuffer, width: Int, height: Int): Boolean
    private external fun nativeGenerateResponse(prompt: String, maxTokens: Int): String
    private external fun nativeFreeModels()

    /**
     * Load all ExecuTorch models and tokenizer
     * @param modelDirPath Directory containing .pte model files
     * @param tokenizerPath Path to tokenizer.json
     * @return true if all models loaded successfully
     */
    suspend fun loadModels(modelDirPath: String, tokenizerPath: String): Boolean {
        return withContext(runLoop) {
            Log.d(tag, "Loading NanoVLM models from: $modelDirPath")
            Log.d(tag, "Loading tokenizer from: $tokenizerPath")
            val success = nativeLoadModels(modelDirPath, tokenizerPath)
            if (success) {
                Log.d(tag, "NanoVLM models loaded successfully")
            } else {
                Log.e(tag, "Failed to load NanoVLM models")
            }
            success
        }
    }

    /**
     * Unload all models and free memory
     */
    suspend fun unloadModels() {
        withContext(runLoop) {
            Log.d(tag, "Unloading NanoVLM models")
            nativeFreeModels()
        }
    }

    /**
     * Process an image from Android Bitmap
     * Converts to ARGB8888 buffer and passes to native code
     * @param bitmap Input image
     * @return true if image processed successfully
     */
    fun processImage(bitmap: Bitmap): Boolean {
        // Ensure bitmap is in ARGB_8888 format
        val config = if (bitmap.config == Bitmap.Config.ARGB_8888) {
            bitmap
        } else {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        }

        val byteBuffer = ByteBuffer.allocateDirect(config.byteCount)
        config.copyPixelsToBuffer(byteBuffer)
        byteBuffer.rewind()

        Log.d(tag, "Processing image: ${config.width}x${config.height}")
        val success = nativeProcessImageFromBuffer(byteBuffer, config.width, config.height)
        if (success) {
            Log.d(tag, "Image processed successfully")
        } else {
            Log.e(tag, "Failed to process image")
        }
        return success
    }

    /**
     * Generate text response for a prompt with the current image context
     * Returns a Flow that emits generated tokens as they're produced
     * @param prompt Text prompt for the model
     * @param maxTokens Maximum number of tokens to generate
     * @return Flow of generated text tokens
     */
    fun generateResponse(prompt: String, maxTokens: Int = 100): Flow<String> = callbackFlow {
        withContext(runLoop) {
            try {
                Log.d(tag, "Generating response for prompt: '$prompt' (max $maxTokens tokens)")

                // ExecuTorch inference is synchronous, so we generate and send the full response
                val response = nativeGenerateResponse(prompt, maxTokens)

                Log.d(tag, "Generation complete. Response length: ${response.length}")

                // Send the full response
                trySend(response).onFailure { exception: Throwable? ->
                    Log.e(tag, "Failed to send response to flow", exception)
                    close(exception)
                }

                // Mark generation as complete
                close()

            } catch (e: Exception) {
                Log.e(tag, "Exception in generateResponse", e)
                cancel("Error: ${e.message}", e)
            }
        }

        awaitClose {
            Log.d(tag, "Generation flow closed")
        }
    }

    /**
     * Check if models are loaded
     * Note: This is a simple check - you might want to add a native method for this
     */
    var isLoaded: Boolean = false
        private set

    suspend fun setLoaded(loaded: Boolean) {
        withContext(runLoop) {
            isLoaded = loaded
        }
    }

    companion object {
        // Enforce only one instance of NanoVLM_Android (singleton)
        private val _instance: NanoVLM_Android = NanoVLM_Android()
        fun instance(): NanoVLM_Android = _instance
    }
}
