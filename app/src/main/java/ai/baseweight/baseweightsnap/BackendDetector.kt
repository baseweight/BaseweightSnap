package ai.baseweight.baseweightsnap

import android.content.Context
import android.os.Build
import android.util.Log

object BackendDetector {
    private const val TAG = "BackendDetector"
    private const val PREFS_NAME = "backend_prefs"
    private const val KEY_BACKEND = "gpu_backend"

    /**
     * Returns the preferred GPU backend, taking into account the user's
     * saved preference. Falls back to auto-detection if set to "auto"
     * or no preference is saved.
     */
    fun getPreferredGpuBackend(context: Context): String {
        val saved = getSavedBackend(context)
        if (saved != "auto") {
            Log.d(TAG, "Using saved backend preference: $saved")
            return saved
        }
        return getAutoDetectedBackend()
    }

    /**
     * Auto-detect backend without reading preferences.
     * Used during initial setup or when preference is "auto".
     */
    fun getAutoDetectedBackend(): String {
        val isQualcomm = isQualcommDevice()
        Log.d(TAG, "Device: ${Build.MANUFACTURER} / ${Build.HARDWARE}, Qualcomm=$isQualcomm")
        return if (isQualcomm) "opencl" else "vulkan"
    }

    fun getSavedBackend(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_BACKEND, "auto") ?: "auto"
    }

    fun saveBackend(context: Context, backend: String) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_BACKEND, backend).apply()
        Log.i(TAG, "Saved backend preference: $backend")
    }

    fun isQualcommDevice(): Boolean {
        // API 31+ has SOC_MANUFACTURER
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val socManufacturer = Build.SOC_MANUFACTURER
            Log.d(TAG, "SOC_MANUFACTURER: $socManufacturer")
            if (socManufacturer.equals("Qualcomm", ignoreCase = true)) {
                return true
            }
        }

        // Fallback: check HARDWARE and BOARD for known Qualcomm identifiers
        val hw = Build.HARDWARE.lowercase()
        val board = Build.BOARD.lowercase()
        val qualcommHints = listOf("qcom", "qualcomm", "msm", "sdm", "sm8", "sm7", "sm6", "snapdragon")
        return qualcommHints.any { hint -> hw.contains(hint) || board.contains(hint) }
    }
}
