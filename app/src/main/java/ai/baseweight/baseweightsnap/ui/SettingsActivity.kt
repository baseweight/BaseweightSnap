package ai.baseweight.baseweightsnap.ui

import ai.baseweight.baseweightsnap.BackendDetector
import ai.baseweight.baseweightsnap.R
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.widget.ImageView
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.card.MaterialCardView

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        val toolbar = findViewById<androidx.appcompat.widget.Toolbar>(R.id.toolbar)
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        toolbar.setNavigationOnClickListener { finish() }

        val rgBackend = findViewById<RadioGroup>(R.id.rgBackend)
        val rbAuto = findViewById<RadioButton>(R.id.rbAuto)
        val rbVulkan = findViewById<RadioButton>(R.id.rbVulkan)
        val rbOpenCL = findViewById<RadioButton>(R.id.rbOpenCL)
        val rbHexagon = findViewById<RadioButton>(R.id.rbHexagon)
        val tvBackendNote = findViewById<TextView>(R.id.tvBackendNote)
        val tvBackendDescription = findViewById<TextView>(R.id.tvBackendDescription)
        val tvDeviceInfo = findViewById<TextView>(R.id.tvDeviceInfo)
        val tvActiveBackend = findViewById<TextView>(R.id.tvActiveBackend)

        val isQualcomm = BackendDetector.isQualcommDevice()

        if (!isQualcomm) {
            // Non-Qualcomm: lock to Vulkan, disable all options
            rbAuto.isEnabled = false
            rbVulkan.isEnabled = false
            rbOpenCL.isEnabled = false
            rbHexagon.isEnabled = false
            rbVulkan.isChecked = true
            tvBackendDescription.text = "Vulkan is the only supported backend on this device"
        } else {
            // Qualcomm: show all options, load saved preference
            val saved = BackendDetector.getSavedBackend(this)
            when (saved) {
                "vulkan" -> rbVulkan.isChecked = true
                "opencl" -> rbOpenCL.isChecked = true
                "hexagon" -> rbHexagon.isChecked = true
                else -> rbAuto.isChecked = true
            }

            rgBackend.setOnCheckedChangeListener { _, checkedId ->
                val backend = when (checkedId) {
                    R.id.rbVulkan -> "vulkan"
                    R.id.rbOpenCL -> "opencl"
                    R.id.rbHexagon -> "hexagon"
                    else -> "auto"
                }
                BackendDetector.saveBackend(this, backend)
                tvBackendNote.visibility = android.view.View.VISIBLE
                updateActiveBackendText(tvActiveBackend)
            }
        }

        // Device info
        val deviceInfo = buildString {
            append("${Build.MANUFACTURER} ${Build.MODEL}")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                append("\nSoC: ${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}")
            }
            append("\nHardware: ${Build.HARDWARE}")
            append("\nAndroid ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
        }
        tvDeviceInfo.text = deviceInfo
        updateActiveBackendText(tvActiveBackend)

        // Manage Models card
        findViewById<MaterialCardView>(R.id.cardManageModels).setOnClickListener {
            startActivity(Intent(this, ModelManagerActivity::class.java))
        }
    }

    private fun updateActiveBackendText(tv: TextView) {
        val active = BackendDetector.getPreferredGpuBackend(this)
        tv.text = "Active backend: $active"
    }
}
