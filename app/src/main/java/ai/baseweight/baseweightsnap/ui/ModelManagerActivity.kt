package ai.baseweight.baseweightsnap.ui

import ai.baseweight.baseweightsnap.ModelManager
import ai.baseweight.baseweightsnap.R
import ai.baseweight.baseweightsnap.models.HFModelMetadata
import ai.baseweight.baseweightsnap.models.ValidationResult
import ai.baseweight.baseweightsnap.services.ModelDownloadService
import android.app.AlertDialog
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
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
                // Could show details dialog in the future
                Toast.makeText(this, "Model: ${model.name}", Toast.LENGTH_SHORT).show()
            },
            onSetDefault = { model ->
                setModelAsDefault(model)
            },
            onDelete = { model ->
                showDeleteConfirmation(model)
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
        // Refresh list when returning to activity
        loadModels()
    }

    private fun loadModels() {
        val models = modelManager.listDownloadedModels()

        if (models.isEmpty()) {
            rvModels.visibility = View.GONE
            emptyStateView.visibility = View.VISIBLE
        } else {
            rvModels.visibility = View.VISIBLE
            emptyStateView.visibility = View.GONE
            modelAdapter.submitList(models)
        }
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
        lifecycleScope.launch {
            // Show validating message
            Toast.makeText(
                this@ModelManagerActivity,
                R.string.validating_repository,
                Toast.LENGTH_SHORT
            ).show()

            // Validate repository
            val validation = modelManager.validateHFRepo(repo)

            when (validation) {
                is ValidationResult.Valid -> {
                    // Start download service
                    ModelDownloadService.startDownload(this@ModelManagerActivity, repo)

                    Toast.makeText(
                        this@ModelManagerActivity,
                        R.string.downloading_model,
                        Toast.LENGTH_SHORT
                    ).show()

                    // Close activity to show notification
                    finish()
                }
                is ValidationResult.Invalid -> {
                    val errorMessage = when (validation.error) {
                        ai.baseweight.baseweightsnap.models.ValidationError.REPO_NOT_FOUND ->
                            "Repository not found"
                        ai.baseweight.baseweightsnap.models.ValidationError.MISSING_CONFIG ->
                            "Missing config.json"
                        ai.baseweight.baseweightsnap.models.ValidationError.MISSING_LANGUAGE_MODEL ->
                            "Missing language model (GGUF file)"
                        ai.baseweight.baseweightsnap.models.ValidationError.MISSING_VISION_MODEL ->
                            "Missing vision model (mmproj GGUF)"
                        ai.baseweight.baseweightsnap.models.ValidationError.INVALID_REPO_FORMAT ->
                            "Invalid repository format"
                        else -> "Validation failed"
                    }

                    AlertDialog.Builder(this@ModelManagerActivity)
                        .setTitle("Validation Failed")
                        .setMessage(errorMessage)
                        .setPositiveButton("OK", null)
                        .show()
                }
            }
        }
    }

    private fun setModelAsDefault(model: HFModelMetadata) {
        modelManager.setDefaultModel(model.id)
        Toast.makeText(this, R.string.model_set_as_default, Toast.LENGTH_SHORT).show()
        loadModels() // Refresh to update UI
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
