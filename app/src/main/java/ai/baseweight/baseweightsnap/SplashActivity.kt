package ai.baseweight.baseweightsnap

import android.content.Intent
import android.os.Bundle
import android.view.WindowManager
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import ai.baseweight.baseweightsnap.databinding.ActivitySplashBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.content.Context
import java.io.File

class SplashActivity : AppCompatActivity() {
    private lateinit var binding: ActivitySplashBinding
    private val scope = CoroutineScope(Dispatchers.Main)
    private val vlmRunner: SmolVLMAndroid = SmolVLMAndroid.instance()
    private val modelManager: ModelManager by lazy { ModelManager(this) }

    private val DEFAULT_ERROR_MESSAGE = "An unexpected error occurred. Please try reinstalling the app."
    private val TAG = "BaseweightSplash"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Set full screen
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        )
        
        binding = ActivitySplashBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Start the model loading process
        checkAndLoadModels()
    }

    private fun checkAndLoadModels() {
        scope.launch {
            try {
                // Check if models are already downloaded
                val defaultModelName = ModelManager.DEFAULT_MODEL_NAME
                Log.d(TAG, "Checking if models are downloaded for: $defaultModelName")
                val modelsDownloaded = modelManager.isModelSetDownloaded(defaultModelName)
                Log.d(TAG, "Models downloaded: $modelsDownloaded")

                if (modelsDownloaded) {
                    Log.d(TAG, "Models are downloaded, attempting to load...")
                    binding.splashStatus.text = "Loading models..."
                    if (loadModels()) {
                        Log.d(TAG, "Models loaded successfully, starting main activity")
                        delay(500)
                        startMainActivity()
                    } else {
                        Log.e(TAG, "Failed to load models into memory")
                        binding.splashStatus.text = "Failed to load models. Please try again."
                        binding.splashProgress.visibility = View.GONE
                    }
                } else {
                    Log.d(TAG, "Models not downloaded, checking WiFi...")
                    // Check if on WiFi
                    val isOnWifi = isOnWiFi()
                    if (isOnWifi) {
                        downloadAndLoadModels()
                    } else {
                        showWiFiWarningDialog()
                    }
                }
            } catch (e: Exception) {
                binding.splashStatus.text = "Error: ${e.message}"
                binding.splashProgress.visibility = View.GONE
            }
        }
    }

    private suspend fun downloadAndLoadModels() {
        binding.splashStatus.text = "Downloading models..."
        try {
            modelManager.downloadModelSet(ModelManager.DEFAULT_MODEL_NAME)
                .collect { progress ->
                    // Update progress display
                    binding.splashStatus.text = progress.message

                    // Only load models when ALL downloads are complete (this is the final COMPLETED status)
                    if (progress.status == DownloadStatus.COMPLETED && progress.message.contains("All models downloaded")) {
                        Log.d(TAG, "All models downloaded successfully, loading into memory...")
                        binding.splashStatus.text = "Loading models..."

                        val modelSet = modelManager.getModelSet(ModelManager.DEFAULT_MODEL_NAME)!!
                        val modelDir = File(modelManager.getModelPath(modelSet.visionEncoderId)).parent!!
                        val tokenizerPath = modelManager.getTokenizerPath(modelSet.tokenizerId)

                        if (File(modelDir).exists() && File(tokenizerPath).exists()) {
                            if (loadModels()) {
                                delay(500)
                                startMainActivity()
                            } else {
                                Log.e(TAG, "Failed to load models from: $modelDir and $tokenizerPath")
                                showErrorAndExit("Failed to load models after download. Please try reinstalling the app.")
                            }
                        } else {
                            Log.e(TAG, "Model files not found: $modelDir or $tokenizerPath")
                            showErrorAndExit("Download failed: Model files not found. Please try reinstalling the app.")
                        }
                    } else if (progress.status == DownloadStatus.ERROR) {
                        Log.e(TAG, "Download error: ${progress.message}")
                        showErrorAndExit("Download failed: ${progress.message}")
                    }
                }
        } catch (e: Exception) {
            val errorMessage = e.message ?: DEFAULT_ERROR_MESSAGE
            Log.e(TAG, "Error downloading models: $errorMessage", e)
            showErrorAndExit("Error downloading models: $errorMessage")
        }
    }

    private fun showErrorAndExit(message: String) {
        scope.launch(Dispatchers.Main) {
            // Log the error
            Log.e(TAG, message)
            
            // Show error dialog
            AlertDialog.Builder(this@SplashActivity)
                .setTitle("Error")
                .setMessage(message)
                .setPositiveButton("OK") { _, _ ->
                    // Close the app
                    finishAffinity()
                    System.exit(0)
                }
                .setCancelable(false)
                .show()
        }
    }

    private fun showWiFiWarningDialog() {
        AlertDialog.Builder(this)
            .setTitle("Large File Download Warning")
            .setMessage("These are large files (total ~2.4GB). Please connect to WiFi for the best experience.\n\nWould you like to continue downloading anyway?")
            .setPositiveButton("Continue") { _, _ ->
                scope.launch {
                    downloadAndLoadModels()
                }
            }
            .setNegativeButton("Exit") { _, _ ->
                finishAffinity() // Close the app
            }
            .setCancelable(false)
            .show()
    }

    private fun isOnWiFi(): Boolean {
        val connectivityManager = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = connectivityManager.activeNetwork ?: return false
        val networkCapabilities = connectivityManager.getNetworkCapabilities(network) ?: return false
        return networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)
    }

    private suspend fun loadModels(): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                val modelSet = modelManager.getModelSet(ModelManager.DEFAULT_MODEL_NAME)!!
                val modelDir = File(modelManager.getModelPath(modelSet.visionEncoderId)).parent!!
                val tokenizerPath = modelManager.getTokenizerPath(modelSet.tokenizerId)

                // Verify files exist before loading
                if (!File(modelDir).exists() || !File(tokenizerPath).exists()) {
                    Log.e(TAG, "Model files not found: $modelDir or $tokenizerPath")
                    return@withContext false
                }

                Log.d(TAG, "Model paths - modelDir: $modelDir, tokenizerPath: $tokenizerPath")

                // Verify files exist before loading
                val modelDirExists = File(modelDir).exists()
                val tokenizerExists = File(tokenizerPath).exists()
                Log.d(TAG, "File existence - modelDir: $modelDirExists, tokenizer: $tokenizerExists")

                // Load the models
                Log.d(TAG, "Calling vlmRunner.loadModels()...")
                val success = vlmRunner.loadModels(modelDir, tokenizerPath)
                Log.d(TAG, "vlmRunner.loadModels() returned: $success")

                if (!success) {
                    Log.e(TAG, "Failed to load models from: $modelDir and $tokenizerPath")
                }
                
                success
            } catch (e: Exception) {
                Log.e(TAG, "Error loading models: ${e.message}", e)
                false
            }
        }
    }

    private fun startMainActivity() {
        val intent = Intent(this, MainActivity::class.java)
        startActivity(intent)
        finish()
    }
} 