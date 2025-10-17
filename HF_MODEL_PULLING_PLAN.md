# HuggingFace Model Pulling - Implementation Plan

## Overview
Add support for downloading VLM models directly from HuggingFace repositories. Users can enter a HF repository in the format `orgName/repository`, and the app will automatically detect, download, and load the required model files.

## Current State
- ✅ Models stored in external storage: `context.getExternalFilesDir(null)/models/`
- ✅ Existing download infrastructure with progress tracking (Flow-based)
- ✅ Model pair concept (language + vision) already implemented
- ✅ Background downloads on IO dispatcher
- ✅ WiFi detection for large downloads
- ✅ Auto-download SmolVLM2 on first launch

## Architecture

```
┌─────────────────────────────────────────┐
│         Model Manager Screen            │
│  - List downloaded models               │
│  - Add new model (HF repo input)        │
│  - Delete models                        │
│  - Set default model                    │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│      HuggingFaceApiClient               │
│  - Fetch repo metadata                  │
│  - List files in repo                   │
│  - Validate required files              │
│  - Generate download URLs               │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│      Enhanced ModelManager              │
│  - Store model metadata (JSON)          │
│  - Download from HF repos               │
│  - Validate downloads                   │
│  - Manage multiple models               │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│   DownloadService (Foreground Service)  │
│  - Background downloads                 │
│  - Notification progress                │
│  - Resume on failure                    │
└─────────────────────────────────────────┘
```

## Phase 1: Core Infrastructure

### 1.1 HuggingFaceApiClient.kt
API client for interacting with HuggingFace API.

**Methods:**
- `getRepoInfo(repo: String): Result<HFRepoInfo>` - Get repository metadata
- `listFiles(repo: String): Result<List<HFFile>>` - List all files in repo
- `findRequiredFiles(repo: String): Result<HFModelFiles>` - Find config.json, mmproj GGUF, main GGUF
- `validateModel(repo: String): Result<ValidationResult>` - Validate all required files exist

**HuggingFace API Endpoints:**
```
GET https://huggingface.co/api/models/{orgName}/{repository}
GET https://huggingface.co/api/models/{orgName}/{repository}/tree/main
Download: https://huggingface.co/{orgName}/{repository}/resolve/main/{filename}
```

**Data Classes:**
```kotlin
data class HFRepoInfo(
    val id: String,
    val author: String,
    val modelId: String,
    val sha: String,
    val lastModified: String,
    val private: Boolean,
    val gated: Boolean
)

data class HFFile(
    val path: String,
    val size: Long,
    val type: String  // "file" or "directory"
)

data class HFModelFiles(
    val configFile: HFFile,
    val languageFile: HFFile,
    val visionFile: HFFile
)

enum class ValidationResult {
    VALID,
    MISSING_CONFIG,
    MISSING_LANGUAGE_MODEL,
    MISSING_VISION_MODEL,
    REPO_NOT_FOUND,
    NETWORK_ERROR
}
```

**File Detection Strategy:**
- Config: Look for `config.json`
- Main GGUF: Look for files matching `*.gguf` excluding `mmproj-*`
- Vision GGUF: Look for files matching `mmproj-*.gguf`

### 1.2 Model Metadata Storage
JSON-based storage for model metadata.

**Data Class:**
```kotlin
data class HFModelMetadata(
    val id: String,              // Generated UUID
    val name: String,            // Display name (from repo or user-provided)
    val hfRepo: String,          // "orgName/repository"
    val languageFile: String,    // Filename of main GGUF
    val visionFile: String,      // Filename of mmproj GGUF
    val configFile: String,      // "config.json"
    val downloadDate: Long,      // Timestamp
    val isDefault: Boolean,      // Is this the default model?
    val languageSize: Long,      // Size in bytes
    val visionSize: Long         // Size in bytes
)

data class ModelMetadataStore(
    val models: List<HFModelMetadata>,
    val defaultModelId: String?
)
```

**Storage Location:**
- `models/metadata.json`

**Operations:**
```kotlin
class ModelMetadataManager(context: Context) {
    fun saveMetadata(metadata: HFModelMetadata)
    fun getAllModels(): List<HFModelMetadata>
    fun getModelById(id: String): HFModelMetadata?
    fun deleteModel(id: String)
    fun setDefaultModel(id: String)
    fun getDefaultModel(): HFModelMetadata?
}
```

## Phase 2: Enhanced ModelManager

### 2.1 New Methods in ModelManager.kt

```kotlin
class ModelManager(private val context: Context) {
    // Existing methods...

    // NEW: Download model from HuggingFace repository
    suspend fun downloadFromHuggingFace(
        repo: String,
        onProgress: (DownloadProgress) -> Unit
    ): Result<HFModelMetadata>

    // NEW: List all downloaded models (including HF models)
    fun listDownloadedModels(): List<HFModelMetadata>

    // NEW: Delete a model and its files
    suspend fun deleteModel(modelId: String): Result<Unit>

    // NEW: Set a model as default
    fun setDefaultModel(modelId: String): Result<Unit>

    // NEW: Get the current default model
    fun getDefaultModel(): HFModelMetadata?

    // NEW: Validate HF repository before downloading
    suspend fun validateHFRepo(repo: String): ValidationResult

    // NEW: Get model paths for HF model
    fun getHFModelPaths(modelId: String): Pair<String, String>? // (language, vision)
}
```

### 2.2 Download Flow for HF Models

1. User enters repository: `orgName/repository`
2. Validate repository via HF API
3. Find required files (config.json, main GGUF, mmproj GGUF)
4. Check available storage space
5. Create model metadata entry
6. Download files with progress tracking
7. Verify downloaded files
8. Update metadata on completion
9. Offer to set as default model

## Phase 3: Background Download Service

### 3.1 ModelDownloadService.kt
Foreground service for downloading models in the background.

**Features:**
- Persistent notification showing download progress
- Handles app being in background
- Survives configuration changes
- Support for cancellation
- Retry logic for failed downloads

**Service Methods:**
```kotlin
class ModelDownloadService : Service() {
    companion object {
        fun startDownload(context: Context, repo: String)
        fun cancelDownload(context: Context)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int
    private fun downloadModel(repo: String)
    private fun updateNotification(progress: Int, message: String)
    private fun onDownloadComplete(modelId: String)
    private fun onDownloadError(error: String)
}
```

**Notification Channels:**
- Channel ID: `model_downloads`
- Importance: DEFAULT
- Show progress bar with percentage
- Action: Cancel download

## Phase 4: UI Components

### 4.1 ModelManagerActivity.kt
New activity for managing downloaded models.

**Features:**
- RecyclerView displaying all downloaded models
- FAB for adding new model from HF
- Swipe-to-delete functionality
- Context menu: Set as default, Delete, View details
- Empty state: "No models downloaded"
- Default model indicator (star icon)

**UI Elements:**
```xml
<!-- activity_model_manager.xml -->
- Toolbar with title "Manage Models"
- RecyclerView for model list
- FloatingActionButton for "Add Model"
- Empty state view
```

### 4.2 Model List Item Layout
```xml
<!-- item_model.xml -->
- Model name (TextView)
- Repository (TextView, small, gray)
- Size information (TextView)
- Download date (TextView)
- Default indicator (ImageView, star)
- More options menu (ImageButton)
```

### 4.3 Add Model Dialog
```xml
<!-- dialog_add_model.xml -->
- EditText for HF repository (orgName/repository)
- Validation hint
- Examples: "HuggingFaceTB/SmolVLM-Instruct"
- Cancel / Download buttons
```

### 4.4 ModelAdapter.kt
RecyclerView adapter for model list.

```kotlin
class ModelAdapter(
    private val models: List<HFModelMetadata>,
    private val onModelClick: (HFModelMetadata) -> Unit,
    private val onSetDefault: (HFModelMetadata) -> Unit,
    private val onDelete: (HFModelMetadata) -> Unit
) : RecyclerView.Adapter<ModelAdapter.ViewHolder>()
```

### 4.5 MainActivity Integration
Add menu item to launch Model Manager:

```kotlin
override fun onCreateOptionsMenu(menu: Menu): Boolean {
    menuInflater.inflate(R.menu.main_menu, menu)
    return true
}

override fun onOptionsItemSelected(item: MenuItem): Boolean {
    return when (item.itemId) {
        R.id.action_manage_models -> {
            startActivity(Intent(this, ModelManagerActivity::class.java))
            true
        }
        else -> super.onOptionsItemSelected(item)
    }
}
```

## Phase 5: Integration

### 5.1 Update SplashActivity.kt
Load default model instead of hardcoded SmolVLM2.

**Changes:**
```kotlin
private fun checkAndLoadModels() {
    scope.launch {
        val defaultModel = modelManager.getDefaultModel()

        if (defaultModel == null) {
            // No models downloaded, download SmolVLM2 as before
            downloadDefaultModel()
        } else {
            // Load the default model
            loadModel(defaultModel)
        }
    }
}

private suspend fun loadModel(metadata: HFModelMetadata) {
    val (languagePath, visionPath) = modelManager.getHFModelPaths(metadata.id)!!

    if (File(languagePath).exists() && File(visionPath).exists()) {
        vlmRunner.loadModels(languagePath, visionPath)
        startMainActivity()
    } else {
        // Files missing, show error
        showErrorDialog("Model files not found. Please re-download the model.")
    }
}
```

### 5.2 Migration Strategy
For users who already have SmolVLM2 downloaded:

1. On first launch with new version, check for existing models
2. Create metadata entry for existing SmolVLM2
3. Set as default model
4. Continue normal flow

```kotlin
private fun migrateExistingModels() {
    // Check if old SmolVLM2 files exist
    val oldLanguagePath = "${getModelsDirectory()}/smolvlm2-256m-language.gguf"
    val oldVisionPath = "${getModelsDirectory()}/smolvlm2-256m-vision.gguf"

    if (File(oldLanguagePath).exists() && File(oldVisionPath).exists()) {
        // Create metadata for existing model
        val metadata = HFModelMetadata(
            id = "smolvlm2-256m-legacy",
            name = "SmolVLM2-256M-VidInstruct",
            hfRepo = "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF",
            languageFile = "SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
            visionFile = "mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
            configFile = "config.json",
            downloadDate = System.currentTimeMillis(),
            isDefault = true,
            languageSize = File(oldLanguagePath).length(),
            visionSize = File(oldVisionPath).length()
        )

        metadataManager.saveMetadata(metadata)
        metadataManager.setDefaultModel(metadata.id)
    }
}
```

## Phase 6: HuggingFace OAuth (Stretch Goal)

### 6.1 Overview
Add authentication support for gated/private models using OAuth 2.0.

### 6.2 Components

**HFAuthManager.kt:**
```kotlin
class HFAuthManager(private val context: Context) {
    fun startOAuthFlow(activity: Activity)
    suspend fun exchangeCodeForToken(code: String): Result<HFToken>
    suspend fun refreshToken(): Result<HFToken>
    suspend fun getUserInfo(): Result<HFUser>
    fun isAuthenticated(): Boolean
    fun logout()
}
```

**HFTokenStorage.kt:**
```kotlin
class HFTokenStorage(context: Context) {
    // Uses EncryptedSharedPreferences
    fun saveToken(token: HFToken)
    fun getToken(): HFToken?
    fun clearToken()
}
```

**HFAccountActivity.kt:**
- Login with HuggingFace button
- Display logged-in user info
- Logout option
- Token status indicator

### 6.3 OAuth Flow
1. User clicks "Login with HuggingFace"
2. Open browser with authorization URL
3. User authorizes app on HF
4. Redirect back to app with authorization code
5. Exchange code for access token
6. Store token securely
7. Use token for authenticated requests

**OAuth Endpoints:**
```
Authorization: https://huggingface.co/oauth/authorize
Token: https://huggingface.co/oauth/token
User Info: https://huggingface.co/api/whoami-v2
```

### 6.4 Security
- Use `EncryptedSharedPreferences` for token storage
- Implement PKCE for OAuth
- Generate random `state` for CSRF protection
- Never log tokens
- Clear tokens on logout
- Handle token refresh before expiration

### 6.5 UI Changes
- Show lock icon for gated models
- Prompt login when accessing gated model
- "Account" menu item in ModelManagerActivity
- Handle 401/403 errors gracefully

## Error Handling

### Common Error Scenarios
1. **Repository not found** - Show error, don't attempt download
2. **Missing required files** - Show which files are missing
3. **Network errors** - Retry logic with exponential backoff
4. **Insufficient storage** - Calculate required space, warn user
5. **Invalid GGUF files** - Validate file headers after download
6. **Download interrupted** - Clean up partial files, allow retry

### User-Facing Messages
- Clear, actionable error messages
- Suggestions for resolution
- Option to retry or cancel
- Don't expose technical details unnecessarily

## Testing Strategy

### Unit Tests
- HuggingFaceApiClient responses
- Metadata storage operations
- File detection logic
- Validation rules

### Integration Tests
- Full download flow
- Model loading after download
- Model deletion and cleanup
- Default model switching

### Manual Testing
- Download various VLM models from HF
- Test with slow network
- Test with interrupted downloads
- Test storage full scenario
- Test model switching
- Test with gated models (Phase 6)

## File Structure

```
app/src/main/java/ai/baseweight/baseweightsnap/
├── models/
│   ├── HuggingFaceApiClient.kt      [NEW]
│   ├── HFModelMetadata.kt           [NEW]
│   ├── ModelMetadataManager.kt      [NEW]
│   └── ValidationResult.kt          [NEW]
├── services/
│   └── ModelDownloadService.kt      [NEW]
├── ui/
│   ├── ModelManagerActivity.kt      [NEW]
│   ├── ModelAdapter.kt              [NEW]
│   └── AddModelDialog.kt            [NEW]
├── auth/ (Phase 6)
│   ├── HFAuthManager.kt             [NEW]
│   ├── HFTokenStorage.kt            [NEW]
│   ├── HFAccountActivity.kt         [NEW]
│   └── OAuthCallbackActivity.kt     [NEW]
├── ModelManager.kt                  [MODIFIED]
├── SplashActivity.kt                [MODIFIED]
└── MainActivity.kt                  [MODIFIED]

app/src/main/res/
├── layout/
│   ├── activity_model_manager.xml   [NEW]
│   ├── item_model.xml               [NEW]
│   ├── dialog_add_model.xml         [NEW]
│   └── activity_hf_account.xml      [NEW - Phase 6]
├── menu/
│   └── main_menu.xml                [NEW]
└── values/
    └── strings.xml                  [MODIFIED]
```

## Dependencies

```kotlin
// Add to build.gradle.kts
dependencies {
    // For JSON parsing
    implementation("com.google.code.gson:gson:2.10.1")

    // For encrypted storage (Phase 6)
    implementation("androidx.security:security-crypto:1.1.0-alpha06")

    // For CustomTabs (Phase 6 OAuth)
    implementation("androidx.browser:browser:1.7.0")
}
```

## Success Criteria

### Phase 1-5 (Core Features)
- ✅ Users can download models from any public HF repo
- ✅ App validates required files exist before downloading
- ✅ Downloads happen in background with progress notification
- ✅ Users can manage multiple models
- ✅ Users can set any model as default
- ✅ Users can delete unused models
- ✅ App loads default model on startup
- ✅ Existing SmolVLM2 users migrated seamlessly

### Phase 6 (OAuth - Stretch Goal)
- ✅ Users can log in with HuggingFace account
- ✅ Users can access gated models
- ✅ Tokens stored securely
- ✅ Token refresh handled automatically
- ✅ Users can log out

## Timeline Estimate

- **Phase 1**: 2-3 hours (API client + metadata)
- **Phase 2**: 2-3 hours (Enhanced ModelManager)
- **Phase 3**: 2-3 hours (Download service)
- **Phase 4**: 3-4 hours (UI components)
- **Phase 5**: 1-2 hours (Integration + testing)
- **Phase 6**: 4-5 hours (OAuth - optional)

**Total**: 10-15 hours for Phases 1-5, +4-5 hours for Phase 6

## Notes

- Keep SmolVLM2 as the default model for new users
- Ensure backward compatibility with existing installations
- All model files stored in external storage
- Support for offline model management (list, delete, switch)
- OAuth is stretch goal - implement only if needed
