package ai.baseweight.baseweightsnap

import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class MTMD_Android {
    private val tag: String? = this::class.simpleName

    private val runLoop: CoroutineDispatcher = Executors.newSingleThreadExecutor {
        thread(start = false, name = "Llm-RunLoop") {
            Log.d(tag, "Dedicated thread for native code: ${Thread.currentThread().name}")

            // No-op if called more than once.
            System.loadLibrary("baseweightsnap")

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
    private external fun generate_response(prompt: String, max_tokens: Int): String
    private external fun get_token_count(text: String): Int
    private external fun stop_generation()
    private external fun reset_stop_flag()

    suspend fun loadModels(languageModelPath: String, mmprojPath: String): Boolean {
        return withContext(runLoop) {
            load_models(languageModelPath, mmprojPath)
        }
    }

    suspend fun unloadModels() {
        withContext(runLoop) {
            free_models()
        }
    }

    fun processImage(bitmap: Bitmap): Boolean {
        // Save bitmap to a temporary file
        val tempFile = File.createTempFile("image", ".png")
        try {
            FileOutputStream(tempFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            return process_image(tempFile.absolutePath)
        } finally {
            tempFile.delete()
        }
    }

    fun generateResponse(prompt: String, maxTokens: Int): Flow<String> = flow {
        reset_stop_flag()  // Reset before starting
        try {
            val response = generate_response(prompt, maxTokens)
            emit(response)
        } finally {
            // Ensure we reset the flag even if generation is interrupted
            reset_stop_flag()
        }
    }.flowOn(runLoop)

    suspend fun getTokenCount(text: String): Int {
        return withContext(runLoop) {
            get_token_count(text)
        }
    }

    fun stopGeneration() {
        stop_generation()
    }

    companion object {
        // Enforce only one instance of MTMD_Android
        private val _instance: MTMD_Android = MTMD_Android()
        fun instance(): MTMD_Android = _instance
    }
}
