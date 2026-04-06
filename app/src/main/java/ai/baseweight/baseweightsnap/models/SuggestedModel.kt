package ai.baseweight.baseweightsnap.models

/**
 * A recommended model that users can download with one tap.
 * All models are <= 4B parameters to work well on mobile devices.
 */
data class SuggestedModel(
    val name: String,
    val hfRepo: String,
    val description: String,
    val parameterSize: String,
    val approximateDownloadSize: String
) {
    companion object {
        val SUGGESTED_MODELS = listOf(
            SuggestedModel(
                name = "SmolVLM2 256M Video Instruct",
                hfRepo = "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF",
                description = "Ultra-compact vision model, fast on any device",
                parameterSize = "256M",
                approximateDownloadSize = "~280 MB"
            ),
            SuggestedModel(
                name = "SmolVLM2 2.2B Instruct",
                hfRepo = "ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
                description = "Best quality-to-size ratio for mobile vision tasks",
                parameterSize = "2.2B",
                approximateDownloadSize = "~1.7 GB"
            ),
            SuggestedModel(
                name = "Gemma 3 4B Instruct",
                hfRepo = "ggml-org/gemma-3-4b-it-GGUF",
                description = "Google's multimodal model with unified architecture",
                parameterSize = "4B",
                approximateDownloadSize = "~3.3 GB"
            ),
            SuggestedModel(
                name = "Gemma 4 E4B Instruct",
                hfRepo = "unsloth/gemma-4-E4B-it-GGUF",
                description = "Google's latest multimodal model with expert mixture",
                parameterSize = "4B",
                approximateDownloadSize = "~3.4 GB"
            ),
            SuggestedModel(
                name = "LFM2-VL 450M",
                hfRepo = "LiquidAI/LFM2-VL-450M-GGUF",
                description = "Liquid AI's ultra-lightweight vision model",
                parameterSize = "450M",
                approximateDownloadSize = "~300 MB"
            ),
        )
    }
}
