package ai.baseweight.baseweightsnap

import android.os.Build
import android.util.Log

object BackendDetector {
    private const val TAG = "BackendDetector"

    /**
     * Returns the preferred GPU backend for the current device.
     *
     * - Qualcomm (Adreno GPU): "opencl" — Vulkan crashes when CLIP vision
     *   and LLM share the Adreno GPU. OpenCL is fast and stable.
     * - Everything else (Pixel/Tensor, etc.): "vulkan"
     */
    fun getPreferredGpuBackend(): String {
        val isQualcomm = isQualcommDevice()
        Log.d(TAG, "Device: ${Build.MANUFACTURER} / ${Build.HARDWARE}, Qualcomm=$isQualcomm")
        return if (isQualcomm) "opencl" else "vulkan"
    }

    private fun isQualcommDevice(): Boolean {
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
