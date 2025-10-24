package ai.baseweight.baseweightsnap.ui

import ai.baseweight.baseweightsnap.ModelManager
import ai.baseweight.baseweightsnap.R
import ai.baseweight.baseweightsnap.models.DownloadState
import ai.baseweight.baseweightsnap.models.HFModelMetadata
import ai.baseweight.baseweightsnap.models.ValidationResult
import ai.baseweight.baseweightsnap.services.ModelDownloadService
import android.Manifest
import android.app.AlertDialog
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.textfield.TextInputEditText
import kotlinx.coroutines.launch

class ModelManagerActivity : AppCompatActivity() {

    private lateinit var modelManager: ModelManager
    private lateinit var modelAdapter: ModelAdapter
    private lateinit var rvModels: RecyclerView
    private lateinit var emptyStateView: View
    private lateinit var fabAddModel: FloatingActionButton

    // Track pending downloads
    private val pendingDownloads = mutableMapOf<String, HFModelMetadata>()

    // Track repo waiting for notification permission
    private var pendingDownloadRepo: String? = null

    // Notification permission launcher
    private val requestNotificationPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            val repo = pendingDownloadRepo ?: return@registerForActivityResult
            pendingDownloadRepo = null

            if (!isGranted) {
                // Show warning but proceed with download
                showNotificationPermissionDeniedDialog(repo)
            } else {
                // Permission granted, proceed with download
                proceedWithDownload(repo)
            }
        }

    // Broadcast receiver for download progress
    private val downloadProgressReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val repo = intent.getStringExtra("repo") ?: return
            val progress = intent.getIntExtra("progress", 0)
            val status = intent.getStringExtra("status") ?: return

            android.util.Log.d("ModelManagerActivity", "Received broadcast: repo=$repo, progress=$progress, status=$status")
            updateDownloadProgress(repo, progress, status)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_model_manager)

        // Initialize ModelManager
        modelManager = ModelManager(this)

        // Setup toolbar
        val toolbar = findViewById<androidx.appcompat.widget.Toolbar>(R.id.toolbar)
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        toolbar.setNavigationOnClickListener {
            finish()
        }

        // Initialize views
        rvModels = findViewById(R.id.rvModels)
        emptyStateView = findViewById(R.id.emptyStateView)
        fabAddModel = findViewById(R.id.fabAddModel)

        // Setup RecyclerView
        modelAdapter = ModelAdapter(
            onModelClick = { model ->
                // Tapping a model switches to it (sets as default)
                if (model.downloadState == DownloadState.COMPLETED || model.downloadState == null) {
                    setModelAsDefault(model)
                }
            },
            onSetDefault = { model ->
                setModelAsDefault(model)
            },
            onDelete = { model ->
                showDeleteConfirmation(model)
            },
            onCancelDownload = { model ->
                cancelDownload(model)
            }
        )

        rvModels.apply {
            layoutManager = LinearLayoutManager(this@ModelManagerActivity)
            adapter = modelAdapter
        }

        // Setup FAB
        fabAddModel.setOnClickListener {
            showAddModelDialog()
        }

        // Load models
        loadModels()
    }

    override fun onResume() {
        super.onResume()
        // Register broadcast receiver for download progress
        val filter = IntentFilter("ai.baseweight.baseweightsnap.DOWNLOAD_PROGRESS")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(downloadProgressReceiver, filter, RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(downloadProgressReceiver, filter)
        }

        // Clean up stale pending downloads (in case downloads completed while app was backgrounded)
        cleanupStalePendingDownloads()

        // Refresh list when returning to activity
        loadModels()
    }

    override fun onPause() {
        super.onPause()
        // Unregister broadcast receiver
        try {
            unregisterReceiver(downloadProgressReceiver)
        } catch (e: Exception) {
            // Already unregistered
        }
    }

    private fun cleanupStalePendingDownloads() {
        // Get list of downloaded models
        val downloadedModels = modelManager.listDownloadedModels()
        val downloadedRepos = downloadedModels.map { it.hfRepo }.toSet()

        // Remove any pending downloads that are now completed
        val iterator = pendingDownloads.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (downloadedRepos.contains(entry.key)) {
                android.util.Log.d("ModelManagerActivity", "Removing stale pending download: ${entry.key}")
                iterator.remove()
            }
        }
    }

    private fun loadModels() {
        val models = modelManager.listDownloadedModels().toMutableList()

        // Add pending downloads to the list
        models.addAll(pendingDownloads.values)

        if (models.isEmpty()) {
            rvModels.visibility = View.GONE
            emptyStateView.visibility = View.VISIBLE
        } else {
            rvModels.visibility = View.VISIBLE
            emptyStateView.visibility = View.GONE
            modelAdapter.submitList(models)
        }
    }

    private fun updateDownloadProgress(repo: String, progress: Int, status: String) {
        val pending = pendingDownloads[repo]
        if (pending != null) {
            val updated = when (status) {
                "DOWNLOADING" -> pending.copy(
                    downloadState = DownloadState.DOWNLOADING,
                    downloadProgress = progress
                )
                "COMPLETED" -> {
                    // Remove from pending and refresh to show completed model
                    pendingDownloads.remove(repo)
                    loadModels()
                    return
                }
                "CANCELLED" -> {
                    // Remove from pending
                    pendingDownloads.remove(repo)
                    Toast.makeText(this, "Download cancelled", Toast.LENGTH_SHORT).show()
                    loadModels()
                    return
                }
                "ERROR" -> pending.copy(
                    downloadState = DownloadState.ERROR,
                    downloadProgress = 0
                )
                else -> pending
            }
            pendingDownloads[repo] = updated
            loadModels()
        }
    }

    private fun cancelDownload(model: HFModelMetadata) {
        // Request service to cancel download
        ModelDownloadService.cancelDownload(this)

        // Remove from pending downloads
        pendingDownloads.remove(model.hfRepo)
        loadModels()

        Toast.makeText(this, "Cancelling download...", Toast.LENGTH_SHORT).show()
    }

    private fun showAddModelDialog() {
        val dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_add_model, null)
        val etRepository = dialogView.findViewById<TextInputEditText>(R.id.etRepository)

        // Example click listeners
        dialogView.findViewById<View>(R.id.tvExample1).setOnClickListener {
            etRepository.setText("HuggingFaceTB/SmolVLM-Instruct")
        }

        dialogView.findViewById<View>(R.id.tvExample2).setOnClickListener {
            etRepository.setText("ggml-org/SmolVLM2-256M-Video-Instruct-GGUF")
        }

        val dialog = MaterialAlertDialogBuilder(this)
            .setView(dialogView)
            .create()

        dialogView.findViewById<View>(R.id.btnCancel).setOnClickListener {
            dialog.dismiss()
        }

        dialogView.findViewById<View>(R.id.btnDownload).setOnClickListener {
            val repo = etRepository.text?.toString()?.trim()

            if (repo.isNullOrEmpty() || !repo.contains("/")) {
                Toast.makeText(
                    this,
                    R.string.invalid_repository_format,
                    Toast.LENGTH_SHORT
                ).show()
                return@setOnClickListener
            }

            dialog.dismiss()
            startDownload(repo)
        }

        dialog.show()
    }

    private fun startDownload(repo: String) {
        // Check notification permission on Android 13+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val notificationPermission = ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.POST_NOTIFICATIONS
            )

            if (notificationPermission != PackageManager.PERMISSION_GRANTED) {
                // Store repo and request permission
                pendingDownloadRepo = repo
                requestNotificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
                return
            }
        }

        // Permission granted or not needed (older Android), proceed with download
        proceedWithDownload(repo)
    }

    private fun proceedWithDownload(repo: String) {
        // Create a pending model entry
        val pendingModel = HFModelMetadata(
            id = "pending_$repo",
            name = repo.substringAfter("/"),
            hfRepo = repo,
            languageFile = "",
            visionFile = "",
            languageSize = 0,
            visionSize = 0,
            downloadState = DownloadState.PENDING,
            downloadProgress = 0
        )
        pendingDownloads[repo] = pendingModel

        // Refresh UI to show pending model
        loadModels()

        // Start download service immediately (it will validate in background)
        ModelDownloadService.startDownload(this@ModelManagerActivity, repo)

        Toast.makeText(
            this@ModelManagerActivity,
            "Download started",
            Toast.LENGTH_SHORT
        ).show()

        // Don't finish() - stay on the Model Manager screen
        // The download happens in background service
        // Progress will be shown in the model list
    }

    private fun showNotificationPermissionDeniedDialog(repo: String) {
        AlertDialog.Builder(this)
            .setTitle("Notification Permission Denied")
            .setMessage("Without notification permission, you won't be notified of download completion if you put the app in the background.\n\nDo you want to continue downloading?")
            .setPositiveButton("Continue") { _, _ ->
                proceedWithDownload(repo)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun setModelAsDefault(model: HFModelMetadata) {
        lifecycleScope.launch {
            try {
                // Get the MTMD singleton
                val vlmRunner = ai.baseweight.baseweightsnap.MTMD_Android.instance(this@ModelManagerActivity)

                // Show loading message
                Toast.makeText(
                    this@ModelManagerActivity,
                    "Switching to ${model.name}...",
                    Toast.LENGTH_SHORT
                ).show()

                // Load the new model in IO context (just like SplashActivity does)
                val success = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                    // Get the model paths
                    val paths = modelManager.getHFModelPaths(model.id)
                    if (paths == null) {
                        android.util.Log.e("ModelManagerActivity", "Failed to get paths for model: ${model.id}")
                        return@withContext false
                    }

                    val (languagePath, visionPath) = paths

                    // Verify files exist
                    if (!java.io.File(languagePath).exists() || !java.io.File(visionPath).exists()) {
                        android.util.Log.e("ModelManagerActivity", "Model files not found: $languagePath or $visionPath")
                        return@withContext false
                    }

                    // Unload current models first
                    vlmRunner.unloadModels()

                    // Load the new models
                    vlmRunner.loadModels(languagePath, visionPath)
                }

                if (success) {
                    // Set as default in metadata
                    modelManager.setDefaultModel(model.id)
                    Toast.makeText(
                        this@ModelManagerActivity,
                        "Switched to ${model.name}",
                        Toast.LENGTH_SHORT
                    ).show()
                    loadModels() // Refresh to update UI
                } else {
                    Toast.makeText(
                        this@ModelManagerActivity,
                        "Failed to load model",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            } catch (e: Exception) {
                android.util.Log.e("ModelManagerActivity", "Error switching model: ${e.message}", e)
                Toast.makeText(
                    this@ModelManagerActivity,
                    "Error switching model: ${e.message}",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun showDeleteConfirmation(model: HFModelMetadata) {
        AlertDialog.Builder(this)
            .setTitle(R.string.delete_model)
            .setMessage(R.string.delete_confirmation)
            .setPositiveButton("Delete") { _, _ ->
                deleteModel(model)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun deleteModel(model: HFModelMetadata) {
        lifecycleScope.launch {
            val result = modelManager.deleteModel(model.id)

            if (result.isSuccess) {
                Toast.makeText(this@ModelManagerActivity, R.string.model_deleted, Toast.LENGTH_SHORT).show()
                loadModels() // Refresh list
            } else {
                Toast.makeText(
                    this@ModelManagerActivity,
                    "Failed to delete model: ${result.exceptionOrNull()?.message}",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }
}
