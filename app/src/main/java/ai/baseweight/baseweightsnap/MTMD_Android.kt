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

class MTMD_Android(private val context: android.content.Context) {
    private val tag: String? = this::class.simpleName

    private val runLoop: CoroutineDispatcher = Executors.newSingleThreadExecutor {
        thread(start = false, name = "Llm-RunLoop") {
            Log.d(tag, "Dedicated thread for native code: ${Thread.currentThread().name}")

            // Dynamically load the appropriate library based on Vulkan support
            val libraryName = VulkanDetector.getLibraryName(context)
            Log.i(tag, "Loading native library: $libraryName")

            // No-op if called more than once.
            System.loadLibrary(libraryName)

            // Set llama log handler to Android
            log_to_android()
            backend_init(false)

            Log.d(tag, system_info())

            it.run()
        }.apply {
            uncaughtExceptionHandler = Thread.UncaughtExceptionHandler { _, exception: Throwable ->
                Log.e(tag, "Unhandled exception", exception)
            }
        }
    }.asCoroutineDispatcher()

    private val nlen: Int = 128

    private external fun log_to_android()
    private external fun backend_init(numa: Boolean)
    private external fun backend_free()
    private external fun system_info(): String
    private external fun load_models(languageModelPath: String, mmprojPath: String): Boolean
    private external fun free_models()
    private external fun process_image(image_path: String): Boolean
    private external fun process_image_from_byteBuff(arr: ByteBuffer, width: Int, height: Int): Boolean
    private external fun generate_response(
        prompt: String,
        max_tokens: Int,
        callback: TextGenerationCallback
    ): String
    private external fun get_token_count(text: String): Int
    private external fun stop_generation()
    private external fun reset_stop_flag()

    suspend fun loadModels(languageModelPath: String, mmprojPath: String): Boolean {
        return withContext(runLoop) {
            load_models(languageModelPath, mmprojPath)
        }
    }

    fun stopGeneration() {
        stop_generation()
    }

    suspend fun unloadModels() {
        withContext(runLoop) {
            free_models()
        }
    }

    fun processImage(bitmap: Bitmap): Boolean {
        // Ensure bitmap is in ARGB_8888 format
        val config = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap else bitmap.copy(Bitmap.Config.ARGB_8888, false)
        
        val byteBuffer = ByteBuffer.allocateDirect(config.byteCount)
        config.copyPixelsToBuffer(byteBuffer)
        byteBuffer.rewind()
        return process_image_from_byteBuff(byteBuffer, config.width, config.height)
    }

    fun generateResponse(prompt: String, maxTokens: Int): Flow<String> = callbackFlow {
        withContext(runLoop) {
            reset_stop_flag()  // Reset before starting

            val callback = object : TextGenerationCallback {
                override fun onTextGenerated(text: String) {
                    trySend(text).onFailure {
                        exception: Throwable? ->
                        Log.e(tag, "Failed to send text to flow", exception)
                        stop_generation()
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
                    trySend("PROGRESS:$phase:$progress").onFailure {
                        exception: Throwable? ->
                        Log.e(tag, "Failed to send progress update", exception)
                    }
                }
            }

            try {
                generate_response(prompt, maxTokens, callback)
            } catch (e: Exception) {
                Log.e(tag, "Exception in generateResponse", e)
                cancel("Error: ${e.message}", null)
            } finally {
                reset_stop_flag()
            }
        }

        awaitClose {
            stop_generation()
        }
    }

    companion object {
        // Enforce only one instance of MTMD_Android
        @Volatile
        private var _instance: MTMD_Android? = null

        fun instance(context: android.content.Context): MTMD_Android {
            return _instance ?: synchronized(this) {
                _instance ?: MTMD_Android(context.applicationContext).also { _instance = it }
            }
        }
    }
}
