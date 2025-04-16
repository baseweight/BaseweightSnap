package ai.baseweight.baseweightsnap

import android.content.Intent
import android.os.Bundle
import android.view.WindowManager
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import ai.baseweight.baseweightsnap.databinding.ActivitySplashBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay

class SplashActivity : AppCompatActivity() {
    private lateinit var binding: ActivitySplashBinding
    private lateinit var modelDownloader: ModelDownloader
    private val scope = CoroutineScope(Dispatchers.Main)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Set full screen
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        )
        
        binding = ActivitySplashBinding.inflate(layoutInflater)
        setContentView(binding.root)

        modelDownloader = ModelDownloader(this)
        downloadModels()
    }

    private fun downloadModels() {
        scope.launch {
            val startTime = System.currentTimeMillis()
            
            modelDownloader.downloadModels { success, errorMessage ->
                if (success) {
                    // Calculate remaining time to reach 1 second
                    val elapsedTime = System.currentTimeMillis() - startTime
                    val remainingTime = maxOf(0, 1000 - elapsedTime)
                    
                    // Launch main activity after the remaining time
                    scope.launch {
                        delay(remainingTime)
                        startMainActivity()
                    }
                } else {
                    binding.splashStatus.text = "Error: $errorMessage"
                    // TODO: Add retry button or error handling
                }
            }
        }
    }

    private fun startMainActivity() {
        val intent = Intent(this, MainActivity::class.java)
        startActivity(intent)
        finish()
    }
} 