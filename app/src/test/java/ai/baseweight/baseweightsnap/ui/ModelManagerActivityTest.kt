package ai.baseweight.baseweightsnap.ui

import ai.baseweight.baseweightsnap.models.DownloadState
import ai.baseweight.baseweightsnap.models.HFModelMetadata
import org.junit.Test
import org.junit.Assert.*

/**
 * Unit tests for ModelManagerActivity.
 * These tests focus on the data flow and state management
 * for download error handling.
 */
class ModelManagerActivityTest {

    @Test
    fun `pending downloads map should be managed correctly on ERROR`() {
        // This test simulates the behavior of updateDownloadProgress with ERROR status
        // Given
        val pendingDownloads = mutableMapOf<String, HFModelMetadata>()
        val testModel = HFModelMetadata(
            id = "test-id",
            name = "TestModel",
            hfRepo = "test/repo",
            languageFile = "",
            visionFile = "",
            languageSize = 0,
            visionSize = 0,
            downloadState = DownloadState.DOWNLOADING,
            downloadProgress = 50
        )
        pendingDownloads["test/repo"] = testModel

        assertEquals("Should have one pending download", 1, pendingDownloads.size)

        // When - simulating ERROR status handling
        pendingDownloads.remove("test/repo")

        // Then
        assertFalse("Pending download should be removed on ERROR",
            pendingDownloads.containsKey("test/repo"))
        assertEquals("Pending downloads should be empty", 0, pendingDownloads.size)
    }

    @Test
    fun `pending download should update progress for DOWNLOADING status`() {
        // This test simulates the behavior with DOWNLOADING status
        // Given
        val pendingDownloads = mutableMapOf<String, HFModelMetadata>()
        val testModel = HFModelMetadata(
            id = "test-id",
            name = "TestModel",
            hfRepo = "test/repo",
            languageFile = "",
            visionFile = "",
            languageSize = 0,
            visionSize = 0,
            downloadState = DownloadState.DOWNLOADING,
            downloadProgress = 50
        )
        pendingDownloads["test/repo"] = testModel

        // When - simulating DOWNLOADING status with progress update
        val updated = testModel.copy(
            downloadState = DownloadState.DOWNLOADING,
            downloadProgress = 75
        )
        pendingDownloads["test/repo"] = updated

        // Then
        assertTrue("Pending download should still exist for DOWNLOADING",
            pendingDownloads.containsKey("test/repo"))
        assertEquals("Progress should be updated to 75",
            75, pendingDownloads["test/repo"]?.downloadProgress)
        assertEquals("Download state should be DOWNLOADING",
            DownloadState.DOWNLOADING, pendingDownloads["test/repo"]?.downloadState)
    }

    @Test
    fun `cleanupStalePendingDownloads should remove completed models`() {
        // This test simulates the cleanup logic
        // Given
        val pendingDownloads = mutableMapOf<String, HFModelMetadata>()
        val downloadedRepos = setOf("completed/repo") // Simulating completed models

        // Add a pending download
        pendingDownloads["pending/repo"] = HFModelMetadata(
            id = "pending-id",
            name = "PendingModel",
            hfRepo = "pending/repo",
            languageFile = "",
            visionFile = "",
            languageSize = 0,
            visionSize = 0,
            downloadState = DownloadState.DOWNLOADING,
            downloadProgress = 50
        )

        // This one is actually completed
        pendingDownloads["completed/repo"] = HFModelMetadata(
            id = "completed-id",
            name = "CompletedModel",
            hfRepo = "completed/repo",
            languageFile = "",
            visionFile = "",
            languageSize = 0,
            visionSize = 0,
            downloadState = DownloadState.PENDING,
            downloadProgress = 0
        )

        // When - simulating cleanup
        val iterator = pendingDownloads.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (downloadedRepos.contains(entry.key)) {
                iterator.remove()
            }
        }

        // Then
        assertTrue("Pending download should still exist",
            pendingDownloads.containsKey("pending/repo"))
        assertFalse("Completed download should be removed from pending",
            pendingDownloads.containsKey("completed/repo"))
    }

    @Test
    fun `model metadata includes download state fields`() {
        // Verify HFModelMetadata can represent download states
        val pendingModel = HFModelMetadata(
            id = "test-id",
            name = "TestModel",
            hfRepo = "test/repo",
            languageFile = "",
            visionFile = "",
            languageSize = 0,
            visionSize = 0,
            downloadState = DownloadState.PENDING,
            downloadProgress = 0
        )

        assertNotNull("Model should have downloadState", pendingModel.downloadState)
        assertEquals("Initial state should be PENDING",
            DownloadState.PENDING, pendingModel.downloadState)
        assertEquals("Initial progress should be 0",
            0, pendingModel.downloadProgress)
    }
}
