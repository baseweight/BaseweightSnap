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

class SmolVLMAndroid {
    private val tag: String? = this::class.simpleName

    private val runLoop: CoroutineDispatcher = Executors.newSingleThreadExecutor {
        thread(start = false, name = "SmolVLM-RunLoop") {
            Log.d(tag, "Dedicated thread for ONNX Runtime: ${Thread.currentThread().name}")

            // Load the native library directly (pure Rust, no CMake wrapper)
            System.loadLibrary("smolvlm_snap")

            Log.d(tag, "SmolVLM ONNX Runtime initialized")

            it.run()
        }.apply {
            uncaughtExceptionHandler = Thread.UncaughtExceptionHandler { _, exception: Throwable ->
                Log.e(tag, "Unhandled exception", exception)
            }
        }
    }.asCoroutineDispatcher()

    // Native method declarations
    private external fun nativeLoadModels(modelDir: String, tokenizerPath: String): Boolean
    private external fun processImageFromBuffer(buffer: ByteBuffer, width: Int, height: Int): Boolean
    private external fun nativeGenerateResponse(prompt: String, maxTokens: Int): String

    suspend fun loadModels(modelDir: String, tokenizerPath: String): Boolean {
        return withContext(runLoop) {
            nativeLoadModels(modelDir, tokenizerPath)
        }
    }

    fun processImage(bitmap: Bitmap): Boolean {
        // Ensure bitmap is in ARGB_8888 format
        val config = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap else bitmap.copy(Bitmap.Config.ARGB_8888, false)

        val byteBuffer = ByteBuffer.allocateDirect(config.byteCount)
        config.copyPixelsToBuffer(byteBuffer)
        byteBuffer.rewind()
        return processImageFromBuffer(byteBuffer, config.width, config.height)
    }

    fun stopGeneration() {
        // For now, this is a no-op since we don't have streaming generation implemented
        // In a full implementation, this would cancel ongoing generation
        Log.d(tag, "stopGeneration called")
    }

    fun generateResponse(prompt: String, maxTokens: Int): Flow<String> = callbackFlow {
        withContext(runLoop) {
            val callback = object : TextGenerationCallback {
                override fun onTextGenerated(text: String) {
                    trySend(text).onFailure { exception: Throwable? ->
                        Log.e(tag, "Failed to send text to flow", exception)
                        close(exception)
                    }
                }

                override fun onGenerationComplete() {
                    close()
                }

                override fun onGenerationError(error: String) {
                    cancel("Generation error: $error", null)
                }

                override fun onProgressUpdate(phase: String, progress: Int) {
                    trySend("PROGRESS:$phase:$progress").onFailure { exception: Throwable? ->
                        Log.e(tag, "Failed to send progress update", exception)
                    }
                }
            }

            try {
                val result = nativeGenerateResponse(prompt, maxTokens)
                // For now, just send the complete result
                // In a more sophisticated implementation, we'd use streaming
                trySend(result).onFailure { exception: Throwable? ->
                    Log.e(tag, "Failed to send result", exception)
                    close(exception)
                }
                close()
            } catch (e: Exception) {
                Log.e(tag, "Exception in generateResponse", e)
                cancel("Error: ${e.message}", null)
            }
        }

        awaitClose {
            // Cleanup if needed
        }
    }

    companion object {
        // Enforce only one instance of SmolVLMAndroid
        private val _instance: SmolVLMAndroid = SmolVLMAndroid()
        fun instance(): SmolVLMAndroid = _instance
    }
}