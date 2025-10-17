package ai.baseweight.baseweightsnap.models

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class ModelMetadataManagerTest {

    private lateinit var context: Context
    private lateinit var manager: ModelMetadataManager
    private lateinit var metadataFile: File

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        manager = ModelMetadataManager(context)

        // Get reference to metadata file for cleanup
        val modelsDir = File(context.getExternalFilesDir(null), "models")
        metadataFile = File(modelsDir, "metadata.json")
    }

    @After
    fun cleanup() {
        // Clean up test data
        if (metadataFile.exists()) {
            metadataFile.delete()
        }
    }

    @Test
    fun testSaveAndRetrieveMetadata() {
        val metadata = HFModelMetadata(
            name = "Test Model",
            hfRepo = "test/model",
            languageFile = "model.gguf",
            visionFile = "mmproj.gguf",
            languageSize = 100_000_000L,
            visionSize = 50_000_000L
        )

        manager.saveMetadata(metadata)

        val retrieved = manager.getModelById(metadata.id)
        assertNotNull(retrieved)
        assertEquals(metadata.name, retrieved?.name)
        assertEquals(metadata.hfRepo, retrieved?.hfRepo)
    }

    @Test
    fun testGetAllModels() {
        val metadata1 = HFModelMetadata(
            name = "Model 1",
            hfRepo = "test/model1",
            languageFile = "model1.gguf",
            visionFile = "mmproj1.gguf",
            languageSize = 100_000_000L,
            visionSize = 50_000_000L
        )

        val metadata2 = HFModelMetadata(
            name = "Model 2",
            hfRepo = "test/model2",
            languageFile = "model2.gguf",
            visionFile = "mmproj2.gguf",
            languageSize = 200_000_000L,
            visionSize = 100_000_000L
        )

        manager.saveMetadata(metadata1)
        manager.saveMetadata(metadata2)

        val allModels = manager.getAllModels()
        assertEquals(2, allModels.size)
        assertTrue(allModels.any { it.id == metadata1.id })
        assertTrue(allModels.any { it.id == metadata2.id })
    }

    @Test
    fun testDeleteModel() {
        val metadata = HFModelMetadata(
            name = "Test Model",
            hfRepo = "test/model",
            languageFile = "model.gguf",
            visionFile = "mmproj.gguf",
            languageSize = 100_000_000L,
            visionSize = 50_000_000L
        )

        manager.saveMetadata(metadata)
        assertNotNull(manager.getModelById(metadata.id))

        manager.deleteModel(metadata.id)
        assertNull(manager.getModelById(metadata.id))
    }

    @Test
    fun testSetDefaultModel() {
        val metadata = HFModelMetadata(
            name = "Test Model",
            hfRepo = "test/model",
            languageFile = "model.gguf",
            visionFile = "mmproj.gguf",
            languageSize = 100_000_000L,
            visionSize = 50_000_000L
        )

        manager.saveMetadata(metadata)
        manager.setDefaultModel(metadata.id)

        val defaultModel = manager.getDefaultModel()
        assertNotNull(defaultModel)
        assertEquals(metadata.id, defaultModel?.id)
        assertTrue(defaultModel?.isDefault == true)
    }

    @Test
    fun testSetDefaultModelClearsOldDefault() {
        val metadata1 = HFModelMetadata(
            name = "Model 1",
            hfRepo = "test/model1",
            languageFile = "model1.gguf",
            visionFile = "mmproj1.gguf",
            languageSize = 100_000_000L,
            visionSize = 50_000_000L
        )

        val metadata2 = HFModelMetadata(
            name = "Model 2",
            hfRepo = "test/model2",
            languageFile = "model2.gguf",
            visionFile = "mmproj2.gguf",
            languageSize = 200_000_000L,
            visionSize = 100_000_000L
        )

        manager.saveMetadata(metadata1)
        manager.saveMetadata(metadata2)

        manager.setDefaultModel(metadata1.id)
        assertEquals(metadata1.id, manager.getDefaultModel()?.id)

        manager.setDefaultModel(metadata2.id)
        val defaultModel = manager.getDefaultModel()
        assertEquals(metadata2.id, defaultModel?.id)

        // Verify old default is no longer default
        val model1 = manager.getModelById(metadata1.id)
        assertFalse(model1?.isDefault == true)
    }

    @Test
    fun testDeleteDefaultModelClearsDefault() {
        val metadata = HFModelMetadata(
            name = "Test Model",
            hfRepo = "test/model",
            languageFile = "model.gguf",
            visionFile = "mmproj.gguf",
            languageSize = 100_000_000L,
            visionSize = 50_000_000L
        )

        manager.saveMetadata(metadata)
        manager.setDefaultModel(metadata.id)
        assertNotNull(manager.getDefaultModel())

        manager.deleteModel(metadata.id)
        assertNull(manager.getDefaultModel())
    }

    @Test
    fun testHasModels() {
        assertFalse(manager.hasModels())

        val metadata = HFModelMetadata(
            name = "Test Model",
            hfRepo = "test/model",
            languageFile = "model.gguf",
            visionFile = "mmproj.gguf",
            languageSize = 100_000_000L,
            visionSize = 50_000_000L
        )

        manager.saveMetadata(metadata)
        assertTrue(manager.hasModels())
    }

    @Test
    fun testGetTotalSize() {
        val metadata1 = HFModelMetadata(
            name = "Model 1",
            hfRepo = "test/model1",
            languageFile = "model1.gguf",
            visionFile = "mmproj1.gguf",
            languageSize = 100_000_000L,
            visionSize = 50_000_000L
        )

        val metadata2 = HFModelMetadata(
            name = "Model 2",
            hfRepo = "test/model2",
            languageFile = "model2.gguf",
            visionFile = "mmproj2.gguf",
            languageSize = 200_000_000L,
            visionSize = 100_000_000L
        )

        manager.saveMetadata(metadata1)
        manager.saveMetadata(metadata2)

        val expectedTotal = metadata1.totalSize + metadata2.totalSize
        assertEquals(expectedTotal, manager.getTotalSize())
    }

    @Test
    fun testUpdateExistingMetadata() {
        val metadata = HFModelMetadata(
            id = "test-id",
            name = "Original Name",
            hfRepo = "test/model",
            languageFile = "model.gguf",
            visionFile = "mmproj.gguf",
            languageSize = 100_000_000L,
            visionSize = 50_000_000L
        )

        manager.saveMetadata(metadata)

        // Update with same ID
        val updatedMetadata = metadata.copy(name = "Updated Name")
        manager.saveMetadata(updatedMetadata)

        val allModels = manager.getAllModels()
        assertEquals(1, allModels.size)
        assertEquals("Updated Name", allModels[0].name)
    }
}
