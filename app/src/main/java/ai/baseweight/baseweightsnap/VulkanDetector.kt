package ai.baseweight.baseweightsnap

import android.content.Context
import android.content.pm.PackageManager
import android.util.Log

object VulkanDetector {
    private const val TAG = "VulkanDetector"

    /**
     * Detects if the device supports Vulkan 1.2 or higher
     * @param context Application context
     * @return true if Vulkan 1.2+ is supported, false otherwise
     */
    fun hasVulkan12Support(context: Context): Boolean {
        val packageManager = context.packageManager

        // Check for Vulkan feature support
        val hasVulkan = packageManager.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL)

        if (!hasVulkan) {
            Log.d(TAG, "Device does not support Vulkan hardware")
            return false
        }

        // Check for Vulkan version 1.2 (version code 4198400)
        // Vulkan version encoding: major = version >> 22, minor = (version >> 12) & 0x3ff
        // Version 1.2.0 = (1 << 22) | (2 << 12) = 4198400
        val vulkan12Version = 0x00401000 // 1.2.0 in Vulkan version format

        val hasVulkan12 = packageManager.hasSystemFeature(
            PackageManager.FEATURE_VULKAN_HARDWARE_VERSION,
            vulkan12Version
        )

        Log.d(TAG, "Vulkan 1.2+ support: $hasVulkan12")
        return hasVulkan12
    }

    /**
     * Returns the library name to load based on Vulkan support
     * @param context Application context
     * @return "baseweightsnap-vulkan" if Vulkan 1.2+ is supported, "baseweightsnap-cpu" otherwise
     */
    fun getLibraryName(context: Context): String {
        return if (hasVulkan12Support(context)) {
            Log.i(TAG, "Loading Vulkan-enabled library")
            "baseweightsnap-vulkan"
        } else {
            Log.i(TAG, "Loading CPU-optimized library (NEON/KleidiAI)")
            "baseweightsnap-cpu"
        }
    }
}
