package ai.baseweight.baseweightsnap.models

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import java.io.File

/**
 * Manages storage and retrieval of model metadata
 */
class ModelMetadataManager(private val context: Context) {
    companion object {
        private const val TAG = "ModelMetadataManager"
        private const val METADATA_FILENAME = "metadata.json"
        private const val MODELS_DIR = "models"
    }

    private val gson: Gson = GsonBuilder().setPrettyPrinting().create()
    private val metadataFile: File by lazy {
        val modelsDir = File(context.getExternalFilesDir(null), MODELS_DIR)
        modelsDir.mkdirs()
        File(modelsDir, METADATA_FILENAME)
    }

    /**
     * Load the metadata store from disk
     */
    private fun loadStore(): ModelMetadataStore {
        return try {
            if (!metadataFile.exists()) {
                Log.d(TAG, "Metadata file doesn't exist, returning empty store")
                return ModelMetadataStore()
            }

            val json = metadataFile.readText()
            gson.fromJson(json, ModelMetadataStore::class.java)
        } catch (e: Exception) {
            Log.e(TAG, "Error loading metadata store", e)
            ModelMetadataStore()
        }
    }

    /**
     * Save the metadata store to disk
     */
    private fun saveStore(store: ModelMetadataStore) {
        try {
            val json = gson.toJson(store)
            metadataFile.writeText(json)
            Log.d(TAG, "Saved metadata store with ${store.models.size} models")
        } catch (e: Exception) {
            Log.e(TAG, "Error saving metadata store", e)
        }
    }

    /**
     * Save model metadata
     */
    fun saveMetadata(metadata: HFModelMetadata) {
        val store = loadStore()
        val updatedModels = store.models.filter { it.id != metadata.id } + metadata
        saveStore(store.copy(models = updatedModels))
        Log.d(TAG, "Saved metadata for model: ${metadata.name} (${metadata.id})")
    }

    /**
     * Get all models
     */
    fun getAllModels(): List<HFModelMetadata> {
        return loadStore().models
    }

    /**
     * Get model by ID
     */
    fun getModelById(id: String): HFModelMetadata? {
        return loadStore().models.find { it.id == id }
    }

    /**
     * Delete model metadata
     */
    fun deleteModel(id: String) {
        val store = loadStore()
        val updatedModels = store.models.filter { it.id != id }

        // If we're deleting the default model, clear the default
        val newDefaultId = if (store.defaultModelId == id) null else store.defaultModelId

        saveStore(store.copy(
            models = updatedModels,
            defaultModelId = newDefaultId
        ))
        Log.d(TAG, "Deleted metadata for model: $id")
    }

    /**
     * Set a model as default
     */
    fun setDefaultModel(id: String) {
        val store = loadStore()

        // Verify the model exists
        if (store.models.none { it.id == id }) {
            Log.e(TAG, "Cannot set non-existent model as default: $id")
            return
        }

        // Update all models to set isDefault correctly
        val updatedModels = store.models.map { model ->
            model.copy(isDefault = model.id == id)
        }

        saveStore(store.copy(
            models = updatedModels,
            defaultModelId = id
        ))
        Log.d(TAG, "Set default model: $id")
    }

    /**
     * Get the default model
     */
    fun getDefaultModel(): HFModelMetadata? {
        val store = loadStore()
        val defaultId = store.defaultModelId
        return if (defaultId != null) {
            store.models.find { it.id == defaultId }
        } else {
            // Fallback to first model with isDefault=true
            store.models.find { it.isDefault }
        }
    }

    /**
     * Check if any models are downloaded
     */
    fun hasModels(): Boolean {
        return loadStore().models.isNotEmpty()
    }

    /**
     * Get total size of all models
     */
    fun getTotalSize(): Long {
        return loadStore().models.sumOf { it.totalSize }
    }
}
