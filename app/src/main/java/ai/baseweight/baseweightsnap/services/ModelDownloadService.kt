package ai.baseweight.baseweightsnap.services

import ai.baseweight.baseweightsnap.DownloadProgress
import ai.baseweight.baseweightsnap.DownloadStatus
import ai.baseweight.baseweightsnap.ModelManager
import ai.baseweight.baseweightsnap.R
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.catch

/**
 * Foreground service for downloading models in the background.
 * Shows persistent notification with download progress.
 */
class ModelDownloadService : Service() {

    companion object {
        private const val TAG = "ModelDownloadService"

        // Notification
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "model_downloads"
        private const val CHANNEL_NAME = "Model Downloads"

        // Intent actions
        private const val ACTION_START_DOWNLOAD = "ai.baseweight.baseweightsnap.START_DOWNLOAD"
        private const val ACTION_CANCEL_DOWNLOAD = "ai.baseweight.baseweightsnap.CANCEL_DOWNLOAD"

        // Intent extras
        private const val EXTRA_REPO = "repo"

        /**
         * Start downloading a model from HuggingFace
         */
        fun startDownload(context: Context, repo: String) {
            val intent = Intent(context, ModelDownloadService::class.java).apply {
                action = ACTION_START_DOWNLOAD
                putExtra(EXTRA_REPO, repo)
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }

        /**
         * Cancel the current download
         */
        fun cancelDownload(context: Context) {
            val intent = Intent(context, ModelDownloadService::class.java).apply {
                action = ACTION_CANCEL_DOWNLOAD
            }
            context.startService(intent)
        }
    }

    private lateinit var modelManager: ModelManager
    private lateinit var notificationManager: NotificationManager
    private var downloadJob: Job? = null
    private var currentRepo: String? = null  // Track current download repo
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    override fun onCreate() {
        super.onCreate()
        modelManager = ModelManager(this)
        notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        createNotificationChannel()
        Log.d(TAG, "Service created")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_DOWNLOAD -> {
                val repo = intent.getStringExtra(EXTRA_REPO)
                if (repo != null) {
                    startDownload(repo)
                } else {
                    Log.e(TAG, "No repository provided")
                    stopSelf()
                }
            }
            ACTION_CANCEL_DOWNLOAD -> {
                cancelDownload()
            }
        }

        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        downloadJob?.cancel()
        serviceScope.cancel()
        Log.d(TAG, "Service destroyed")
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                CHANNEL_NAME,
                NotificationManager.IMPORTANCE_DEFAULT
            ).apply {
                description = "Shows progress for model downloads"
                setShowBadge(false)
            }
            notificationManager.createNotificationChannel(channel)
        }
    }

    private fun startDownload(repo: String) {
        // Cancel any existing download
        downloadJob?.cancel()

        // Track current repo
        currentRepo = repo

        // Start foreground with initial notification
        val notification = buildNotification(
            progress = 0,
            message = "Starting download...",
            isIndeterminate = true
        )
        startForeground(NOTIFICATION_ID, notification)

        // Start download in coroutine
        downloadJob = serviceScope.launch {
            try {
                modelManager.downloadFromHuggingFace(repo)
                    .catch { e ->
                        Log.e(TAG, "Download error: ${e.message}", e)
                        onDownloadError(e.message ?: "Download failed")
                    }
                    .collect { progress ->
                        Log.d(TAG, "Progress: ${progress.progress}% - ${progress.status} - ${progress.message}")
                        when (progress.status) {
                            DownloadStatus.PENDING -> {
                                updateNotification(
                                    progress = progress.progress,
                                    message = progress.message,
                                    isIndeterminate = true
                                )
                                sendProgressBroadcast(repo, progress.progress, "PENDING")
                            }
                            DownloadStatus.DOWNLOADING -> {
                                updateNotification(
                                    progress = progress.progress,
                                    message = progress.message,
                                    isIndeterminate = false
                                )
                                sendProgressBroadcast(repo, progress.progress, "DOWNLOADING")
                            }
                            DownloadStatus.COMPLETED -> {
                                sendProgressBroadcast(repo, 100, "COMPLETED")
                                onDownloadComplete(repo)
                            }
                            DownloadStatus.ERROR -> {
                                sendProgressBroadcast(repo, 0, "ERROR")
                                onDownloadError(progress.message)
                            }
                        }
                    }
            } catch (e: Exception) {
                Log.e(TAG, "Download exception: ${e.message}", e)
                onDownloadError(e.message ?: "Download failed")
            }
        }
    }

    private fun cancelDownload() {
        Log.d(TAG, "Download cancelled by user")
        downloadJob?.cancel()

        // Send cancelled broadcast with current repo
        currentRepo?.let { repo ->
            sendProgressBroadcast(repo, 0, "CANCELLED")
        }

        // Show cancellation notification
        val notification = buildNotification(
            progress = 0,
            message = "Download cancelled",
            isIndeterminate = false,
            showCancel = false
        )
        notificationManager.notify(NOTIFICATION_ID, notification)

        // Stop service after a delay to show the message
        serviceScope.launch {
            delay(2000)
            stopSelf()
        }
    }

    private fun onDownloadComplete(repo: String) {
        Log.d(TAG, "Download completed: $repo")

        // Show completion notification
        val notification = buildNotification(
            progress = 100,
            message = "Download complete!",
            isIndeterminate = false,
            showCancel = false
        )
        notificationManager.notify(NOTIFICATION_ID, notification)

        // Stop service
        serviceScope.launch {
            delay(2000)
            stopSelf()
        }
    }

    private fun onDownloadError(error: String) {
        Log.e(TAG, "Download error: $error")

        // Show error notification
        val notification = buildNotification(
            progress = 0,
            message = "Download failed: $error",
            isIndeterminate = false,
            showCancel = false
        )
        notificationManager.notify(NOTIFICATION_ID, notification)

        // Stop service
        serviceScope.launch {
            delay(3000)
            stopSelf()
        }
    }

    private fun sendProgressBroadcast(repo: String, progress: Int, status: String) {
        val intent = Intent("ai.baseweight.baseweightsnap.DOWNLOAD_PROGRESS").apply {
            setPackage(packageName)  // Make broadcast explicit for Android 8.0+
            putExtra("repo", repo)
            putExtra("progress", progress)
            putExtra("status", status)
        }
        Log.d(TAG, "Sending broadcast: repo=$repo, progress=$progress, status=$status")
        sendBroadcast(intent)
    }

    private fun updateNotification(
        progress: Int,
        message: String,
        isIndeterminate: Boolean
    ) {
        val notification = buildNotification(progress, message, isIndeterminate)
        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    private fun buildNotification(
        progress: Int,
        message: String,
        isIndeterminate: Boolean,
        showCancel: Boolean = true
    ) = NotificationCompat.Builder(this, CHANNEL_ID)
        .setContentTitle("Downloading Model")
        .setContentText(message)
        .setSmallIcon(android.R.drawable.stat_sys_download)
        .setPriority(NotificationCompat.PRIORITY_DEFAULT)
        .setOngoing(showCancel)
        .setProgress(100, progress, isIndeterminate)
        .apply {
            if (showCancel) {
                // Add cancel action
                val cancelIntent = Intent(this@ModelDownloadService, ModelDownloadService::class.java).apply {
                    action = ACTION_CANCEL_DOWNLOAD
                }
                val cancelPendingIntent = PendingIntent.getService(
                    this@ModelDownloadService,
                    0,
                    cancelIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
                )
                addAction(
                    android.R.drawable.ic_menu_close_clear_cancel,
                    "Cancel",
                    cancelPendingIntent
                )
            }
        }
        .build()
}
