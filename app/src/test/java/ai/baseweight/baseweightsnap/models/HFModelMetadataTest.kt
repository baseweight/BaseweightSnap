package ai.baseweight.baseweightsnap.models

import org.junit.Assert.*
import org.junit.Test

class HFModelMetadataTest {

    @Test
    fun `test total size calculation`() {
        val metadata = HFModelMetadata(
            name = "Test Model",
            hfRepo = "test/model",
            languageFile = "model.gguf",
            visionFile = "mmproj.gguf",
            languageSize = 100_000_000L,
            visionSize = 50_000_000L
        )

        assertEquals(150_000_000L, metadata.totalSize)
    }

    @Test
    fun `test formatSize for MB`() {
        val metadata = HFModelMetadata(
            name = "Test Model",
            hfRepo = "test/model",
            languageFile = "model.gguf",
            visionFile = "mmproj.gguf",
            languageSize = 500_000_000L, // ~477 MB
            visionSize = 100_000_000L    // ~95 MB
        )

        val formatted = metadata.formatSize()
        assertTrue(formatted.contains("MB"))
    }

    @Test
    fun `test formatSize for GB`() {
        val metadata = HFModelMetadata(
            name = "Test Model",
            hfRepo = "test/model",
            languageFile = "model.gguf",
            visionFile = "mmproj.gguf",
            languageSize = 1_500_000_000L, // ~1.4 GB
            visionSize = 500_000_000L      // ~0.5 GB
        )

        val formatted = metadata.formatSize()
        assertTrue(formatted.contains("GB"))
    }

    @Test
    fun `test HFFile isGGUF detection`() {
        val ggufFile = HFFile("model.gguf", 1000L)
        val txtFile = HFFile("readme.txt", 1000L)

        assertTrue(ggufFile.isGGUF)
        assertFalse(txtFile.isGGUF)
    }

    @Test
    fun `test HFFile isMMProj detection`() {
        val mmprojFile = HFFile("mmproj-model.gguf", 1000L)
        val regularGguf = HFFile("model.gguf", 1000L)

        assertTrue(mmprojFile.isMMProj)
        assertFalse(regularGguf.isMMProj)
    }

    @Test
    fun `test HFFile isConfig detection`() {
        val configFile = HFFile("config.json", 1000L)
        val otherFile = HFFile("other.json", 1000L)

        assertTrue(configFile.isConfig)
        assertFalse(otherFile.isConfig)
    }

    @Test
    fun `test HFModelFiles totalSize`() {
        val files = HFModelFiles(
            configFile = HFFile("config.json", 1000L),
            languageFile = HFFile("model.gguf", 100_000L),
            visionFile = HFFile("mmproj.gguf", 50_000L)
        )

        assertEquals(151_000L, files.totalSize)
    }

    @Test
    fun `test ValidationResult Valid`() {
        val files = HFModelFiles(
            configFile = HFFile("config.json", 1000L),
            languageFile = HFFile("model.gguf", 100_000L),
            visionFile = HFFile("mmproj.gguf", 50_000L)
        )

        val result = ValidationResult.Valid(files)
        assertTrue(result is ValidationResult.Valid)
    }

    @Test
    fun `test ValidationResult Invalid`() {
        val result = ValidationResult.Invalid(ValidationError.MISSING_CONFIG)
        assertTrue(result is ValidationResult.Invalid)
        assertEquals(ValidationError.MISSING_CONFIG, (result as ValidationResult.Invalid).error)
    }
}
