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
    private val vlmRunner: MTMD_Android by lazy { MTMD_Android.instance(this) }
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
                val modelsDownloaded = modelManager.isModelPairDownloaded(defaultModelName)

                if (modelsDownloaded) {
                    binding.splashStatus.text = "Loading models..."
                    if (loadModels()) {
                        delay(500)
                        startMainActivity()
                    } else {
                        binding.splashStatus.text = "Failed to load models. Please try again."
                        binding.splashProgress.visibility = View.GONE
                    }
                } else {
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
            modelManager.downloadModelPair(ModelManager.DEFAULT_MODEL_NAME)
                .collect { progress ->
                    binding.splashStatus.text = "Downloading: ${progress.progress}%"
                    if (progress.status == DownloadStatus.COMPLETED) {
                        // Download is complete, load the models
                        val model = modelManager.getMTMDModel(ModelManager.DEFAULT_MODEL_NAME)!!
                        val languagePath = modelManager.getModelPath(model.languageId)
                        val visionPath = modelManager.getModelPath(model.visionId)

                        if (File(languagePath).exists() && File(visionPath).exists()) {
                            if (loadModels()) {
                                delay(500)
                                startMainActivity()
                            } else {
                                showErrorAndExit("Failed to load models after download. Please try reinstalling the app.")
                            }
                        } else {
                            showErrorAndExit("Download failed: Model files not found. Please try reinstalling the app.")
                        }
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
                val model = modelManager.getMTMDModel(ModelManager.DEFAULT_MODEL_NAME)!!
                val modelPath = modelManager.getModelPath(model.languageId)
                val visionPath = modelManager.getModelPath(model.visionId)
                
                // Verify files exist before loading
                if (!File(modelPath).exists() || !File(visionPath).exists()) {
                    Log.e(TAG, "Model files not found: $modelPath or $visionPath")
                    return@withContext false
                }
                
                // Load the models
                val success = vlmRunner.loadModels(modelPath, visionPath)
                
                if (!success) {
                    Log.e(TAG, "Failed to load models from: $modelPath and $visionPath")
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