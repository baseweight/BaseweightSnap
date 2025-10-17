package ai.baseweight.baseweightsnap.models

import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Integration tests for HuggingFaceApiClient
 * These tests make real API calls to HuggingFace and require network access
 */
@RunWith(AndroidJUnit4::class)
class HuggingFaceApiClientTest {

    private lateinit var client: HuggingFaceApiClient

    @Before
    fun setup() {
        client = HuggingFaceApiClient()
    }

    @Test
    fun `test getRepoInfo with valid repo`() = runBlocking {
        val result = client.getRepoInfo("HuggingFaceTB/SmolVLM-Instruct")

        assertTrue(result.isSuccess)
        val repoInfo = result.getOrNull()!!
        assertEquals("HuggingFaceTB/SmolVLM-Instruct", repoInfo.modelId)
        assertFalse(repoInfo.private)
    }

    @Test
    fun `test getRepoInfo with invalid repo`() = runBlocking {
        val result = client.getRepoInfo("invalid/nonexistent-repo-12345")

        assertTrue(result.isFailure)
    }

    @Test
    fun `test listFiles returns files`() = runBlocking {
        val result = client.listFiles("HuggingFaceTB/SmolVLM-Instruct")

        assertTrue(result.isSuccess)
        val files = result.getOrNull()!!
        assertTrue(files.isNotEmpty())

        // Should contain at least some GGUF files
        assertTrue(files.any { it.isGGUF })
    }

    @Test
    fun `test findRequiredFiles with VLM model`() = runBlocking {
        val result = client.findRequiredFiles("HuggingFaceTB/SmolVLM-Instruct")

        assertTrue(result.isSuccess)
        val files = result.getOrNull()!!

        // Verify all required files found
        assertTrue(files.configFile.isConfig)
        assertTrue(files.languageFile.isGGUF)
        assertTrue(files.visionFile.isMMProj)
        assertFalse(files.languageFile.isMMProj)
    }

    @Test
    fun `test findRequiredFiles with non-VLM model fails`() = runBlocking {
        // Try with a text-only model that doesn't have vision components
        val result = client.findRequiredFiles("gpt2")

        assertTrue(result.isFailure)
    }

    @Test
    fun `test validateModel with valid VLM repo`() = runBlocking {
        val result = client.validateModel("HuggingFaceTB/SmolVLM-Instruct")

        assertTrue(result is ValidationResult.Valid)
        val files = (result as ValidationResult.Valid).files
        assertNotNull(files.configFile)
        assertNotNull(files.languageFile)
        assertNotNull(files.visionFile)
    }

    @Test
    fun `test validateModel with invalid repo format`() = runBlocking {
        val result = client.validateModel("invalid-format")

        assertTrue(result is ValidationResult.Invalid)
        assertEquals(
            ValidationError.INVALID_REPO_FORMAT,
            (result as ValidationResult.Invalid).error
        )
    }

    @Test
    fun `test validateModel with nonexistent repo`() = runBlocking {
        val result = client.validateModel("test/nonexistent-repo-12345")

        assertTrue(result is ValidationResult.Invalid)
        assertEquals(
            ValidationError.REPO_NOT_FOUND,
            (result as ValidationResult.Invalid).error
        )
    }

    @Test
    fun `test getDownloadUrl format`() {
        val url = client.getDownloadUrl(
            repo = "test/model",
            filename = "model.gguf"
        )

        assertEquals(
            "https://huggingface.co/test/model/resolve/main/model.gguf",
            url
        )
    }

    @Test
    fun `test getDownloadUrl with spaces`() {
        val url = client.getDownloadUrl(
            repo = "test/model",
            filename = "model file.gguf"
        )

        assertTrue(url.contains("model%20file.gguf"))
        assertFalse(url.contains(" "))
    }

    @Test
    fun `test getDownloadUrl with custom revision`() {
        val url = client.getDownloadUrl(
            repo = "test/model",
            filename = "model.gguf",
            revision = "dev"
        )

        assertTrue(url.contains("/resolve/dev/"))
    }
}
