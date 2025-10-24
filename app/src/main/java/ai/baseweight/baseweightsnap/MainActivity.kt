package ai.baseweight.baseweightsnap

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.TextView.OnEditorActionListener
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.drawable.toBitmap
import ai.baseweight.baseweightsnap.databinding.ActivityMainBinding
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.app.AlertDialog
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Button
import android.text.Spannable
import android.text.SpannableString
import android.text.style.ClickableSpan
import android.text.method.LinkMovementMethod

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private var isPreviewMode = false
    private var latestImageUri: Uri? = null
    private var currentCameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
    private val scope = CoroutineScope(Dispatchers.Main)
    private var generationJob: Job? = null
    private val vlmRunner: MTMD_Android by lazy { MTMD_Android.instance(this) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Set full screen
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        )
        
        // Hide the status bar
        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_FULLSCREEN
            or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
            or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        )
        
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Show instructions dialog
        showInstructionsDialog()

        // Setup button click listeners
        binding.btnCapture.setOnClickListener { captureImage() }
        binding.btnSwitchCamera.setOnClickListener { switchCamera() }
        binding.btnGallery.setOnClickListener { openGallery() }
        binding.btnClosePreview.setOnClickListener { closePreview() }
        binding.btnAddText.setOnClickListener { showTextInput() }
        binding.btnDescribe.setOnClickListener { describeImageWrapper() }
        binding.btnCancelInput.setOnClickListener { hideTextInput() }
        binding.btnSubmitInput.setOnClickListener { submitTextInput() }
        binding.btnDismissResponse.setOnClickListener { hideResponseText() }
        binding.btnSettings.setOnClickListener { openModelManager() }

        // Setup text input
        binding.promptInput.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_DONE) {
                submitTextInput()
                true
            } else {
                false
            }
        }

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.cameraPreview.surfaceProvider)
                }

            // Image capture
            imageCapture = ImageCapture.Builder()
                .build()

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, currentCameraSelector, preview, imageCapture
                )
            } catch (exc: Exception) {
                exc.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun captureImage() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Create time stamped name and MediaStore entry.
        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = android.content.ContentValues().apply {
            put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, name)
            put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        }

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(contentResolver,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues)
            .build()

        // Set up image capture listener
        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    exc.printStackTrace()
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val savedUri = output.savedUri
                    savedUri?.let {
                        latestImageUri = it
                        binding.imagePreview.setImageURI(it)
                        showPreviewMode(true)
                    }
                }
            }
        )
    }

    private fun switchCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Determine the new camera selector
            currentCameraSelector = if (currentCameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
                CameraSelector.DEFAULT_FRONT_CAMERA
            } else {
                CameraSelector.DEFAULT_BACK_CAMERA
            }

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.cameraPreview.surfaceProvider)
                }

            // Image capture
            imageCapture = ImageCapture.Builder()
                .build()

            try {
                // Unbind all use cases
                cameraProvider.unbindAll()
                
                // Bind use cases to the new camera
                cameraProvider.bindToLifecycle(
                    this, currentCameraSelector, preview, imageCapture
                )
            } catch (exc: Exception) {
                exc.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun showResponseText(text: String) {
        binding.responseText.text = text
        binding.responseText.visibility = View.VISIBLE
        binding.btnDismissResponse.visibility = View.VISIBLE
    }


    private fun stopGeneration() {
        generationJob?.cancel()
        vlmRunner.stopGeneration()
    }

    private fun hideResponseText() {
        stopGeneration()
        binding.responseText.text = ""  // Clear the text
        binding.responseText.visibility = View.GONE
        binding.btnDismissResponse.visibility = View.GONE
    }

    private fun showPreviewMode(show: Boolean) {
        isPreviewMode = show
        binding.cameraPreview.visibility = if (show) View.GONE else View.VISIBLE
        binding.imagePreview.visibility = if (show) View.VISIBLE else View.GONE
        
        // Camera mode buttons
        binding.btnCapture.visibility = if (show) View.GONE else View.VISIBLE
        binding.btnSwitchCamera.visibility = if (show) View.GONE else View.VISIBLE
        binding.btnGallery.visibility = if (show) View.GONE else View.VISIBLE
        
        // Preview mode buttons
        binding.btnClosePreview.visibility = if (show) View.VISIBLE else View.GONE
        binding.btnAddText.visibility = if (show) View.VISIBLE else View.GONE
        binding.btnDescribe.visibility = if (show) View.VISIBLE else View.GONE

        // Hide text input and response when switching modes
        binding.promptInputLayout.visibility = View.GONE
        hideResponseText()
    }

    private fun closePreview() {
        showPreviewMode(false)
        binding.imagePreview.setImageURI(null)
        binding.promptInputLayout.visibility = View.GONE
        hideResponseText()
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK)
        intent.type = "image/*"
        startActivityForResult(intent, REQUEST_IMAGE_PICK)
    }

    private fun showTextInput() {
        // Hide bottom buttons
        binding.btnAddText.visibility = View.GONE
        binding.btnDescribe.visibility = View.GONE
        binding.btnClosePreview.visibility = View.GONE
        
        // Show text input
        binding.promptInputLayout.visibility = View.VISIBLE
        binding.promptInput.requestFocus()
        
        // Show keyboard
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        imm.showSoftInput(binding.promptInput, InputMethodManager.SHOW_IMPLICIT)
    }

    private fun hideTextInput() {
        // Show bottom buttons
        binding.btnAddText.visibility = View.VISIBLE
        binding.btnDescribe.visibility = View.VISIBLE
        binding.btnClosePreview.visibility = View.VISIBLE
        
        // Hide text input
        binding.promptInputLayout.visibility = View.GONE
        
        // Hide keyboard
        val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(binding.promptInput.windowToken, 0)
    }

    private fun submitTextInput() {
        val prompt = binding.promptInput.text?.toString() ?: ""
        if (prompt.isNotEmpty()) {
            // Clear and hide the input
            binding.promptInput.text?.clear()
            hideTextInput()

            // Submit the prompt
            scope.launch {
                describeImage(prompt)
            }

        }
    }

    private fun describeImageWrapper() {
        scope.launch {
            describeImage()
        }
    }

    private suspend fun describeImage(prompt: String = "Can you describe this image") {
        if (latestImageUri == null) {
            showResponseText("No image selected")
            return
        }

        // Get the image from the preview
        val bitmap = binding.imagePreview.drawable?.toBitmap()
        if (bitmap == null) {
            showResponseText("Failed to get image data")
            return
        }

        // Create progress dialog
        val progressDialog = AlertDialog.Builder(this).apply {
            setTitle("Processing Image")
            setView(R.layout.dialog_progress)
            setCancelable(false)
        }.create()

        // Show progress dialog before starting processing
        progressDialog.show()

        // Get progress views
        val progressBar = progressDialog.findViewById<ProgressBar>(R.id.progressBar)
        val progressText = progressDialog.findViewById<TextView>(R.id.progressText)

        // Hide text view initially
        binding.responseText.visibility = View.GONE
        binding.btnDismissResponse.visibility = View.GONE

        try {
            vlmRunner.processImage(bitmap)

            Log.d("MainActivity", "Generating response for prompt: $prompt")
            generationJob = scope.launch {
                vlmRunner.generateResponse(prompt, 2048).collect { text ->
                    Log.d("MainActivity", "Received text: $text")
                    withContext(Dispatchers.Main) {
                        if (text.startsWith("PROGRESS:")) {
                            // Handle progress update
                            val parts = text.substring(9).split(":")
                            if (parts.size == 2) {
                                val (phase, progress) = parts
                                progressText?.text = phase
                                progressBar?.progress = progress.toInt()
                            }
                        } else {
                            // Handle generated text
                            Log.d("MainActivity", "Generated text: $text")
                            // Show text view and dismiss dialog when we get actual text
                            progressDialog.dismiss()
                            binding.responseText.visibility = View.VISIBLE
                            binding.btnDismissResponse.visibility = View.VISIBLE
                            if(binding.responseText.text.isEmpty()) {
                                binding.responseText.text = text
                            } else {
                                binding.responseText.append(text)
                            }
                            binding.responseText.invalidate()
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error generating response", e)
            progressDialog.dismiss()
            showResponseText("Error generating response: ${e.message}")
        }
    }

    // Helper extension function to convert Bitmap to ByteBuffer
    private fun Bitmap.toByteBuffer(): ByteBuffer {
        // Create a direct ByteBuffer
        val buffer = ByteBuffer.allocateDirect(this.byteCount)
        // Copy the pixels
        this.copyPixelsToBuffer(buffer)
        // Reset position to 0
        buffer.rewind()
        return buffer
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                finish()
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_PICK && resultCode == RESULT_OK) {
            data?.data?.let { uri ->
                binding.imagePreview.setImageURI(uri)
                latestImageUri = uri
                showPreviewMode(true)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopGeneration()  // Stop any ongoing generation
        cameraExecutor.shutdown()
    }

    private fun openModelManager() {
        val intent = Intent(this, ai.baseweight.baseweightsnap.ui.ModelManagerActivity::class.java)
        startActivity(intent)
    }

    private fun showInstructionsDialog() {
        val dialog = AlertDialog.Builder(this).apply {
            setView(R.layout.dialog_instructions)
            setCancelable(false)
        }.create()

        dialog.show()

        // Set the button click listener after showing the dialog
        dialog.findViewById<Button>(R.id.btnGotIt)?.setOnClickListener {
            dialog.dismiss()
        }

        // Set up the clickable link
        dialog.findViewById<TextView>(R.id.copyrightText)?.let { textView ->
            val spannableString = SpannableString("© 2024 Baseweight Snap. Visit us at baseweight.ai")
            val clickableSpan = object : ClickableSpan() {
                override fun onClick(widget: View) {
                    val intent = Intent(Intent.ACTION_VIEW, Uri.parse("https://baseweight.ai"))
                    startActivity(intent)
                }
            }
            spannableString.setSpan(clickableSpan, spannableString.length - 13, spannableString.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
            textView.text = spannableString
            textView.movementMethod = LinkMovementMethod.getInstance()
        }
    }

    companion object {
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val REQUEST_IMAGE_PICK = 201
    }
}