package ai.baseweight.baseweightsnap

import android.content.Intent
import android.os.Bundle
import android.view.WindowManager
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import ai.baseweight.baseweightsnap.databinding.ActivitySplashBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

class SplashActivity : AppCompatActivity() {
    private lateinit var binding: ActivitySplashBinding
    private val scope = CoroutineScope(Dispatchers.Main)
    private val vlmRunner: MTMD_Android = MTMD_Android.instance()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Set full screen
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        )
        
        binding = ActivitySplashBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Show loading indicator
        binding.splashProgress.visibility = View.VISIBLE
        binding.splashStatus.visibility = View.VISIBLE
        binding.splashStatus.text = "Loading models..."

        // Launch main activity after loading models
        scope.launch {
            try {
                // Load models
                val success = vlmRunner.loadModels(
                    "/data/local/tmp/gguf/smolvlm2-500M-vid-instruct.gguf",
                    "/data/local/tmp/gguf/mmproj-smolvlm2-500M-vid-instruct.gguf"
                )
                
                if (success) {
                    // Add a small delay to ensure smooth transition
                    delay(500)
                    startMainActivity()
                } else {
                    // Handle loading failure
                    binding.splashStatus.text = "Failed to load models. Please try again."
                    binding.splashProgress.visibility = View.GONE
                }
            } catch (e: Exception) {
                // Handle any exceptions
                binding.splashStatus.text = "Error: ${e.message}"
                binding.splashProgress.visibility = View.GONE
            }
        }
    }

    private fun startMainActivity() {
        val intent = Intent(this, MainActivity::class.java)
        startActivity(intent)
        finish()
    }
} 