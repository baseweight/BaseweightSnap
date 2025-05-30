package ai.baseweight.baseweightsnap

interface TextGenerationCallback {
    fun onTextGenerated(text: String)
    fun onGenerationComplete()
    fun onGenerationError(error: String)
    fun onProgressUpdate(phase: String, progress: Int)
}
