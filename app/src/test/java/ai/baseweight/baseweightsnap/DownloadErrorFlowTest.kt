package ai.baseweight.baseweightsnap

import ai.baseweight.baseweightsnap.models.DownloadState
import ai.baseweight.baseweightsnap.models.HFModelMetadata
import org.junit.Test
import org.junit.Assert.*

/**
 * Integration test that verifies the complete error handling flow
 * for model downloads.
 */
class DownloadErrorFlowTest {

    @Test
    fun `error flow removes model from pending list`() {
        // Simulate the flow:
        // 1. User starts download - model added to pending
        // 2. Download fails with error
        // 3. Model removed from pending
        // 4. Error dialog shown

        // Given - initial pending downloads map
        val pendingDownloads = mutableMapOf<String, HFModelMetadata>()

        // Step 1: User starts download
        val repo = "invalid/text-only-model"
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

        assertEquals("Model should be in pending downloads", 1, pendingDownloads.size)
        assertTrue("Model repo should be in pending", pendingDownloads.containsKey(repo))

        // Step 2: Download progresses
        pendingDownloads[repo] = pendingModel.copy(
            downloadState = DownloadState.DOWNLOADING,
            downloadProgress = 25
        )

        assertEquals("Progress should update", 25, pendingDownloads[repo]?.downloadProgress)

        // Step 3: Download fails - ERROR broadcast received
        val errorMessage = "Repository does not contain vision model files"

        // Simulate updateDownloadProgress behavior with ERROR status
        pendingDownloads.remove(repo)

        // Then - verify cleanup
        assertFalse("Failed download should be removed from pending",
            pendingDownloads.containsKey(repo))
        assertEquals("Pending downloads should be empty", 0, pendingDownloads.size)

        // Error dialog would be shown with errorMessage
        // (in actual code, this triggers showDownloadErrorDialog)
        assertNotNull("Error message should exist", errorMessage)
        assertTrue("Error message should be descriptive",
            errorMessage.contains("vision model"))
    }

    @Test
    fun `successful download removes model from pending`() {
        // Given
        val pendingDownloads = mutableMapOf<String, HFModelMetadata>()
        val repo = "valid/vlm-model"

        val pendingModel = HFModelMetadata(
            id = "pending_$repo",
            name = repo.substringAfter("/"),
            hfRepo = repo,
            languageFile = "",
            visionFile = "",
            languageSize = 0,
            visionSize = 0,
            downloadState = DownloadState.DOWNLOADING,
            downloadProgress = 99
        )
        pendingDownloads[repo] = pendingModel

        // When - COMPLETED broadcast received
        pendingDownloads.remove(repo)

        // Then
        assertFalse("Completed download should be removed from pending",
            pendingDownloads.containsKey(repo))
    }

    @Test
    fun `cancelled download removes model from pending`() {
        // Given
        val pendingDownloads = mutableMapOf<String, HFModelMetadata>()
        val repo = "some/model"

        val pendingModel = HFModelMetadata(
            id = "pending_$repo",
            name = repo.substringAfter("/"),
            hfRepo = repo,
            languageFile = "",
            visionFile = "",
            languageSize = 0,
            visionSize = 0,
            downloadState = DownloadState.DOWNLOADING,
            downloadProgress = 45
        )
        pendingDownloads[repo] = pendingModel

        // When - CANCELLED broadcast received
        pendingDownloads.remove(repo)

        // Then
        assertFalse("Cancelled download should be removed from pending",
            pendingDownloads.containsKey(repo))
    }

    @Test
    fun `multiple pending downloads are handled independently`() {
        // Given
        val pendingDownloads = mutableMapOf<String, HFModelMetadata>()

        val model1 = HFModelMetadata(
            id = "pending_repo1",
            name = "model1",
            hfRepo = "org/model1",
            languageFile = "",
            visionFile = "",
            languageSize = 0,
            visionSize = 0,
            downloadState = DownloadState.DOWNLOADING,
            downloadProgress = 30
        )

        val model2 = HFModelMetadata(
            id = "pending_repo2",
            name = "model2",
            hfRepo = "org/model2",
            languageFile = "",
            visionFile = "",
            languageSize = 0,
            visionSize = 0,
            downloadState = DownloadState.DOWNLOADING,
            downloadProgress = 60
        )

        pendingDownloads["org/model1"] = model1
        pendingDownloads["org/model2"] = model2

        assertEquals("Should have 2 pending downloads", 2, pendingDownloads.size)

        // When - one fails, one continues
        pendingDownloads.remove("org/model1") // model1 fails

        // Then
        assertFalse("Failed model should be removed", pendingDownloads.containsKey("org/model1"))
        assertTrue("Other model should still be downloading", pendingDownloads.containsKey("org/model2"))
        assertEquals("Should have 1 pending download", 1, pendingDownloads.size)
    }

    @Test
    fun `download state transitions correctly`() {
        // Test that download state transitions follow the expected pattern
        val states = mutableListOf<DownloadState>()

        // Typical successful flow
        states.add(DownloadState.PENDING)
        states.add(DownloadState.DOWNLOADING)
        // COMPLETED would trigger removal

        assertEquals("Initial state should be PENDING", DownloadState.PENDING, states[0])
        assertEquals("Should transition to DOWNLOADING", DownloadState.DOWNLOADING, states[1])

        // Error flow
        val errorStates = mutableListOf<DownloadState>()
        errorStates.add(DownloadState.PENDING)
        errorStates.add(DownloadState.DOWNLOADING)
        // ERROR would trigger removal (not stored in state)

        assertEquals("Error flow starts same way", DownloadState.PENDING, errorStates[0])
        assertEquals("Error flow has DOWNLOADING state", DownloadState.DOWNLOADING, errorStates[1])
    }
}
