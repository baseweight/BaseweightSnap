package ai.baseweight.baseweightsnap.models

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/**
 * Client for interacting with HuggingFace API
 */
class HuggingFaceApiClient {
    companion object {
        private const val TAG = "HFApiClient"
        private const val BASE_URL = "https://huggingface.co"
        private const val API_BASE_URL = "$BASE_URL/api"
        private const val TIMEOUT_MS = 30000
    }

    /**
     * Get repository information
     */
    suspend fun getRepoInfo(repo: String): Result<HFRepoInfo> {
        return try {
            val url = "$API_BASE_URL/models/$repo"
            val response = makeRequest(url)

            val json = JSONObject(response)
            val repoInfo = HFRepoInfo(
                id = json.getString("id"),
                author = json.optString("author", ""),
                modelId = json.getString("modelId"),
                sha = json.optString("sha"),
                lastModified = json.optString("lastModified"),
                private = json.optBoolean("private", false),
                gated = json.optBoolean("gated", false),
                downloads = json.optLong("downloads", 0)
            )

            Log.d(TAG, "Repository info: $repoInfo")
            Result.success(repoInfo)
        } catch (e: Exception) {
            Log.e(TAG, "Error fetching repo info for $repo", e)
            Result.failure(e)
        }
    }

    /**
     * List all files in a repository
     */
    suspend fun listFiles(repo: String, revision: String = "main"): Result<List<HFFile>> {
        return try {
            val url = "$API_BASE_URL/models/$repo/tree/$revision"
            val response = makeRequest(url)

            val files = mutableListOf<HFFile>()
            val jsonArray = JSONArray(response)

            for (i in 0 until jsonArray.length()) {
                val fileJson = jsonArray.getJSONObject(i)
                val file = HFFile(
                    path = fileJson.getString("path"),
                    size = fileJson.optLong("size", 0),
                    type = fileJson.optString("type", "file")
                )
                files.add(file)
            }

            Log.d(TAG, "Found ${files.size} files in $repo")
            Result.success(files)
        } catch (e: Exception) {
            Log.e(TAG, "Error listing files for $repo", e)
            Result.failure(e)
        }
    }

    /**
     * Find and validate required files for a VLM model
     */
    suspend fun findRequiredFiles(repo: String): Result<HFModelFiles> {
        val filesResult = listFiles(repo)
        if (filesResult.isFailure) {
            return Result.failure(filesResult.exceptionOrNull()!!)
        }

        val files = filesResult.getOrNull()!!
        Log.d(TAG, "Analyzing files: ${files.map { it.path }}")

        // Note: config.json is not required for GGUF model loading with llama.cpp

        // Find vision model (mmproj GGUF)
        val visionFiles = files.filter { it.isMMProj }
        if (visionFiles.isEmpty()) {
            Log.e(TAG, "Missing vision model (mmproj GGUF)")
            return Result.failure(Exception("Missing vision model (mmproj GGUF)"))
        }
        if (visionFiles.size > 1) {
            Log.w(TAG, "Multiple vision models found: ${visionFiles.map { it.path }}")
        }
        val visionFile = visionFiles.first()

        // Find language model (main GGUF, not mmproj)
        val languageFiles = files.filter { it.isGGUF && !it.isMMProj }
        if (languageFiles.isEmpty()) {
            Log.e(TAG, "Missing language model (GGUF)")
            return Result.failure(Exception("Missing language model (GGUF)"))
        }
        if (languageFiles.size > 1) {
            Log.w(TAG, "Multiple language models found: ${languageFiles.map { it.path }}, using first")
        }
        val languageFile = languageFiles.first()

        val modelFiles = HFModelFiles(
            configFile = null, // config.json not needed for GGUF models
            languageFile = languageFile,
            visionFile = visionFile
        )

        Log.d(TAG, "Found required files: language=${languageFile.path}, vision=${visionFile.path}")
        return Result.success(modelFiles)
    }

    /**
     * Validate a HuggingFace repository for VLM compatibility
     */
    suspend fun validateModel(repo: String): ValidationResult {
        // Validate repo format
        if (!repo.matches(Regex("^[\\w.-]+/[\\w.-]+$"))) {
            Log.e(TAG, "Invalid repo format: $repo")
            return ValidationResult.Invalid(ValidationError.INVALID_REPO_FORMAT)
        }

        // Check if repo exists
        val repoInfoResult = getRepoInfo(repo)
        if (repoInfoResult.isFailure) {
            Log.e(TAG, "Repository not found: $repo")
            return ValidationResult.Invalid(ValidationError.REPO_NOT_FOUND)
        }

        // Find required files
        val filesResult = findRequiredFiles(repo)
        if (filesResult.isFailure) {
            val error = when {
                filesResult.exceptionOrNull()?.message?.contains("vision") == true ->
                    ValidationError.MISSING_VISION_MODEL
                filesResult.exceptionOrNull()?.message?.contains("language") == true ->
                    ValidationError.MISSING_LANGUAGE_MODEL
                else -> ValidationError.NETWORK_ERROR
            }
            return ValidationResult.Invalid(error)
        }

        val files = filesResult.getOrNull()!!
        return ValidationResult.Valid(files)
    }

    /**
     * Generate download URL for a file in a repository
     */
    fun getDownloadUrl(repo: String, filename: String, revision: String = "main"): String {
        val encodedFilename = URLEncoder.encode(filename, "UTF-8").replace("+", "%20")
        return "$BASE_URL/$repo/resolve/$revision/$encodedFilename"
    }

    /**
     * Make an HTTP GET request
     */
    private suspend fun makeRequest(url: String): String = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
        val connection = URL(url).openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "GET"
            connection.connectTimeout = TIMEOUT_MS
            connection.readTimeout = TIMEOUT_MS
            connection.setRequestProperty("User-Agent", "BaseweightSnap/1.0")

            val responseCode = connection.responseCode
            if (responseCode != HttpURLConnection.HTTP_OK) {
                val errorStream = connection.errorStream?.bufferedReader()?.use { it.readText() }
                throw Exception("HTTP $responseCode: ${connection.responseMessage}${errorStream?.let { "\n$it" } ?: ""}")
            }

            return@withContext connection.inputStream.bufferedReader().use { it.readText() }
        } finally {
            connection.disconnect()
        }
    }
}
