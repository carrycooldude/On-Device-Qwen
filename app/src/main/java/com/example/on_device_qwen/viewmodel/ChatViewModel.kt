package com.example.on_device_qwen.viewmodel

import android.app.Application
import android.os.Environment
import android.os.SystemClock
import android.net.Uri
import android.provider.OpenableColumns
import com.example.on_device_qwen.inference.ChunkListener
import com.example.on_device_qwen.inference.DolphinNativeEngine
import com.example.on_device_qwen.model.BenchmarkMetrics
import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.NonCancellable
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.on_device_qwen.model.ChatMessage
import com.google.ai.edge.litertlm.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Sealed class representing the various states of the LLM engine lifecycle.
 */
sealed class EngineState {
    data object Idle : EngineState()
    data class Loading(val progress: String) : EngineState()
    data object Ready : EngineState()
    data class Error(val message: String) : EngineState()
}

enum class BackendType(val label: String, val chipText: String) {
    CPU("CPU (XNNPACK)", "CPU"),
    GPU("GPU (OpenCL)", "GPU"),
    NPU("NPU (Qualcomm HTP)", "NPU")
}

/**
 * ViewModel that manages the LiteRT-LM engine lifecycle, conversation state,
 * and chat message history.
 */
@OptIn(ExperimentalApi::class)
class ChatViewModel(application: Application) : AndroidViewModel(application) {

    // ── Engine state ────────────────────────────────────────────────────
    private val _engineState = MutableStateFlow<EngineState>(EngineState.Idle)
    val engineState: StateFlow<EngineState> = _engineState.asStateFlow()

    // ── Messages ────────────────────────────────────────────────────────
    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    // ── Generating flag (blocks double-send) ────────────────────────────
    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    // ── Model Title ────────────────────────────────────────────────────
    private val _modelTitle = MutableStateFlow("Qwen")
    val modelTitle: StateFlow<String> = _modelTitle.asStateFlow()

    // ── Available Models List ──────────────────────────────────────────
    private val _availableModels = MutableStateFlow<List<File>>(emptyList())
    val availableModels: StateFlow<List<File>> = _availableModels.asStateFlow()

    // ── Active Backend ──────────────────────────────────────────────────
    private val _currentBackend = MutableStateFlow(BackendType.CPU)
    val currentBackend: StateFlow<BackendType> = _currentBackend.asStateFlow()

    // ── Vision Model Flag ───────────────────────────────────────────────
    private val _isVisionModel = MutableStateFlow(false)
    val isVisionModel: StateFlow<Boolean> = _isVisionModel.asStateFlow()

    // ── Internals ───────────────────────────────────────────────────────
    private var engine: Engine? = null
    private var conversation: Conversation? = null
    private var currentModelFile: File? = null
    private var dolphinEngine: DolphinNativeEngine? = null
    private val stopDolphin = AtomicBoolean(false)
    private val deferredNativeClose = AtomicBoolean(false)
    private val _isGgufModel = MutableStateFlow(false)
    val isGgufModel: StateFlow<Boolean> = _isGgufModel.asStateFlow()

    /** Search directories where model files might be placed. */
    private val searchDirectories: List<File>
        get() {
            val app = getApplication<Application>()
            return listOfNotNull(
                app.filesDir,
                app.getExternalFilesDir(null),
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
                File("/sdcard/Download"),
                File("/sdcard/Android/data/${app.packageName}/files"),
                File("/data/local/tmp")
            )
        }

    /**
     * Locate the model file on device, initialize the LiteRT-LM engine,
     * and create a conversation session.
     */
    fun initializeEngine(
        explicitModelFile: File? = null,
        requestedBackend: BackendType? = null
    ) {
        if (_engineState.value is EngineState.Loading || _isGenerating.value) {
            return // already loading
        }

        requestedBackend?.let { _currentBackend.value = it }

        viewModelScope.launch {
            _engineState.value = EngineState.Loading("Searching for model files…")

            try {
                // Close existing engine and conversation if re-initializing
                withContext(Dispatchers.IO) {
                    try { conversation?.close() } catch (_: Exception) {}
                    try { engine?.close() } catch (_: Exception) {}
                    try { dolphinEngine?.close() } catch (_: Exception) {}
                    conversation = null
                    engine = null
                    dolphinEngine = null
                }

                // ── Refresh available models list ────────────────────
                val allModels = withContext(Dispatchers.IO) { scanAvailableModels() }
                _availableModels.value = allModels

                // ── Find the model to load ───────────────────────────
                val modelFile = withContext(Dispatchers.IO) {
                    explicitModelFile?.let { prepareReadableFile(it) } ?: currentModelFile ?: findModelFile()
                }
                if (modelFile == null) {
                    _engineState.value = EngineState.Error(
                        buildString {
                            appendLine("Model file not found!")
                            appendLine()
                            appendLine("Import Dolphin GGUF or push a .litertlm model to one of:")
                            searchDirectories.forEach { appendLine("  • ${it.absolutePath}") }
                            appendLine()
                            appendLine("Quick ADB command:")
                            appendLine("  adb push <model>.litertlm /sdcard/Download/")
                            appendLine()
                            appendLine("Then tap 'Retry' below.")
                        }
                    )
                    return@launch
                }

                currentModelFile = modelFile
                _modelTitle.value = modelFile.nameWithoutExtension

                val isGguf = modelFile.extension.equals("gguf", ignoreCase = true)
                _isGgufModel.value = isGguf
                val isVlm = !isGguf && modelFile.nameWithoutExtension.lowercase().contains("vl")
                _isVisionModel.value = isVlm
                if (isGguf) {
                    _currentBackend.value = BackendType.CPU
                    _engineState.value = EngineState.Loading("Loading Dolphin GGUF on CPU…")
                    withContext(Dispatchers.IO) {
                        dolphinEngine = DolphinNativeEngine().also { it.load(modelFile) }
                    }
                    _engineState.value = EngineState.Ready
                    return@launch
                }

                // ── Create EngineConfig & initialize ────────────────
                val backendType = _currentBackend.value
                _engineState.value = EngineState.Loading(
                    "Loading ${modelFile.nameWithoutExtension} (${modelFile.length() / 1_048_576} MB) on ${backendType.chipText}…\nThis may take a moment."
                )

                withContext(Dispatchers.IO) {
                    val backendObj: Backend = when (backendType) {
                        BackendType.CPU -> Backend.CPU()
                        BackendType.GPU -> Backend.GPU()
                        BackendType.NPU -> Backend.NPU(
                            getApplication<Application>().applicationInfo.nativeLibraryDir
                        )
                    }

                    try {
                        val config = if (isVlm) {
                            EngineConfig(
                                modelPath = modelFile.absolutePath,
                                backend = backendObj,
                                maxNumImages = 1,
                                maxNumTokens = 4096
                            )
                        } else {
                            EngineConfig(
                                modelPath = modelFile.absolutePath,
                                backend = backendObj
                            )
                        }
                        engine = Engine(config).also { it.initialize() }
                    } catch (backendErr: Exception) {
                        if (backendType != BackendType.CPU) {
                            // Fallback gracefully to CPU if GPU/NPU is unsupported by this model export
                            _currentBackend.value = BackendType.CPU
                            val fallbackConfig = if (isVlm) {
                                EngineConfig(
                                    modelPath = modelFile.absolutePath,
                                    backend = Backend.CPU(),
                                    maxNumImages = 1,
                                    maxNumTokens = 4096
                                )
                            } else {
                                EngineConfig(
                                    modelPath = modelFile.absolutePath,
                                    backend = Backend.CPU()
                                )
                            }
                            engine = Engine(fallbackConfig).also { it.initialize() }
                        } else {
                            throw backendErr
                        }
                    }
                }

                // ── Create conversation ─────────────────────────────
                _engineState.value = EngineState.Loading("Creating conversation session…")

                withContext(Dispatchers.IO) {
                    val convConfig = ConversationConfig(
                        systemInstruction = Contents.of(SYSTEM_PROMPT)
                    )
                    conversation = engine!!.createConversation(convConfig)
                }

                _engineState.value = EngineState.Ready

            } catch (e: Exception) {
                _engineState.value = EngineState.Error(
                    "Failed to initialize engine:\n${e.localizedMessage ?: e.toString()}"
                )
            }
        }
    }

    /**
     * Switch to a different model on the device.
     */
    fun switchModel(modelFile: File) {
        if (_isGenerating.value) return
        _messages.value = emptyList()
        currentModelFile = modelFile
        initializeEngine(modelFile)
    }

    /**
     * Switch backend (CPU, GPU, NPU).
     */
    fun switchBackend(backend: BackendType) {
        if (_isGenerating.value || _isGgufModel.value) return
        _messages.value = emptyList()
        initializeEngine(requestedBackend = backend)
    }

    /**
     * Scan directories for available Qwen .litertlm files.
     * Excludes any Gemma or incompatible models.
     */
    private fun scanAvailableModels(): List<File> {
        val found = mutableListOf<File>()
        val seenNames = mutableSetOf<String>()

        for (dir in searchDirectories) {
            if (!dir.exists() || !dir.isDirectory) continue
            val files = dir.listFiles { _, name ->
                val lower = name.lowercase()
                (lower.contains("qwen") &&
                !lower.contains("gemma") &&
                !lower.contains("qwen35") &&
                (lower.endsWith(".litertlm") || lower.endsWith(".litertl"))) ||
                (lower.contains("dolphin3.0-qwen2.5-3b") && lower.endsWith(".gguf"))
            } ?: continue

            for (file in files) {
                if (file.length() > 1024 * 1024 && seenNames.add(file.nameWithoutExtension)) {
                    found.add(file)
                }
            }
        }
        return found
    }

    /**
     * Search for any compatible .litertlm or .litertl model file across directories,
     * prioritizing Qwen models and copying to app-private storage if needed for accessibility.
     */
    private fun findModelFile(): File? {
        val app = getApplication<Application>()
        val internalFile = File(app.filesDir, "Qwen2.5-1.5B-Instruct.litertlm")
        if (internalFile.exists() && internalFile.length() > 0) {
            return internalFile
        }

        val qwen3File = File(app.filesDir, "Qwen3-0.6B.litertlm")
        if (qwen3File.exists() && qwen3File.length() > 0) {
            return qwen3File
        }

        val candidates = listOf(
            "Qwen2.5-1.5B-Instruct.litertlm",
            "Qwen3-0.6B.litertlm",
            "Qwen3-0.6B.litertl",
            "Qwen2.5-3B-Instruct.litertlm",
            "qwen.litertlm"
        )

        // 1. Direct candidate matching
        for (dir in searchDirectories) {
            if (!dir.exists() || !dir.isDirectory) continue
            for (candidate in candidates) {
                val file = File(dir, candidate)
                if (file.exists() && file.length() > 0 && file.canRead()) {
                    return prepareReadableFile(file)
                }
            }
        }

        // 2. Scan directories for any file with qwen and .litertl
        for (dir in searchDirectories) {
            if (!dir.exists() || !dir.isDirectory) continue
            val matched = dir.listFiles { _, name ->
                val lower = name.lowercase()
                lower.contains("qwen") &&
                !lower.contains("gemma") &&
                !lower.contains("qwen35") &&
                (lower.endsWith(".litertlm") || lower.endsWith(".litertl"))
            } ?: emptyArray()

            val bestMatch = matched.firstOrNull { it.length() > 0 && it.canRead() }
            if (bestMatch != null) {
                return prepareReadableFile(bestMatch)
            }
        }

        return null
    }

    /**
     * Ensures the file is in an app-accessible location and has the required .litertlm extension.
     */
    private fun prepareReadableFile(source: File): File {
        val app = getApplication<Application>()
        val targetName = if (source.extension.equals("gguf", ignoreCase = true) ||
            source.name.endsWith(".litertlm", ignoreCase = true)) {
            source.name
        } else {
            "${source.nameWithoutExtension}.litertlm"
        }

        if (source.parentFile?.absolutePath == app.filesDir.absolutePath && source.name == targetName) {
            return source
        }
        val dest = File(app.filesDir, targetName)
        if (dest.exists() && dest.length() == source.length()) {
            return dest
        }
        return try {
            _engineState.value = EngineState.Loading("Importing model to app storage…")
            source.inputStream().use { input ->
                dest.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            dest
        } catch (_: Exception) {
            source
        }
    }

    /** Import Dolphin 3.0 GGUF via the document picker, off the UI thread. */
    fun importDolphinGguf(uri: Uri) {
        if (_isGenerating.value || _engineState.value is EngineState.Loading) return
        viewModelScope.launch {
            _engineState.value = EngineState.Loading("Importing Dolphin GGUF to private storage…")
            try {
                val copied = withContext(Dispatchers.IO) {
                    val app = getApplication<Application>()
                    val sourceName = app.contentResolver.query(
                        uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null
                    )?.use { cursor ->
                        if (cursor.moveToFirst()) cursor.getString(0) else null
                    } ?: "Dolphin3.0-Qwen2.5-3b-Q4_K_M.gguf"
                    require(sourceName.matches(Regex(
                        "Dolphin3[.]0-Qwen2[.]5-3b[-_a-zA-Z0-9.]*[.]gguf",
                        RegexOption.IGNORE_CASE
                    ))) { "Select a Dolphin3.0-Qwen2.5-3b GGUF file" }
                    val destination = File(app.filesDir, sourceName)
                    val temporary = File(app.filesDir, "${sourceName}.partial")
                    try {
                        app.contentResolver.openInputStream(uri)?.use { input ->
                            temporary.outputStream().use { output -> input.copyTo(output) }
                        } ?: error("Cannot read selected model file")
                        require(temporary.length() > 1_048_576) { "Incomplete GGUF model" }
                        if (destination.exists()) check(destination.delete()) { "Cannot replace previous model" }
                        check(temporary.renameTo(destination)) { "Unable to finalize model import" }
                        destination
                    } finally {
                        if (temporary.exists()) temporary.delete()
                    }
                }
                _engineState.value = EngineState.Idle
                currentModelFile = copied
                initializeEngine(copied)
            } catch (e: Exception) {
                _engineState.value = EngineState.Error("Dolphin import failed: ${e.localizedMessage}")
            }
        }
    }

    /**
     * Copies an external content URI into a local app cache file for the C++ LiteRT-LM engine.
     */
    private fun saveUriToCache(uri: android.net.Uri): File? {
        return try {
            val app = getApplication<Application>()
            val targetFile = File(app.cacheDir, "vlm_input_${System.currentTimeMillis()}.jpg")
            app.contentResolver.openInputStream(uri)?.use { input ->
                targetFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            targetFile
        } catch (_: Exception) {
            null
        }
    }

    /**
     * Send a user message and stream the model's response token-by-token.
     * Supports multimodal image inputs for Vision-Language Models (VLM).
     * Computes real-time and engine-level TTFT and TPS benchmarking notations.
     */
    fun sendMessage(userText: String, imageUri: android.net.Uri? = null) {
        val trimmed = userText.trim()
        if ((trimmed.isEmpty() && imageUri == null) || _isGenerating.value) return
        if (conversation == null && dolphinEngine == null) return

        val startMs = System.currentTimeMillis()
        var firstTokenMs: Long? = null
        var chunkCount = 0
        val activeBackendName = _currentBackend.value.chipText

        val localImageFile = imageUri?.let { saveUriToCache(it) }
        val promptText = if (trimmed.isNotEmpty()) {
            trimmed
        } else if (localImageFile != null) {
            "Describe this image in detail."
        } else {
            return
        }

        // Add user bubble
        val userMsg = ChatMessage(
            text = promptText,
            imagePath = localImageFile?.absolutePath,
            isUser = true
        )
        _messages.value = _messages.value + userMsg

        // Add placeholder bot bubble with initial benchmarking state
        val botMsgId = System.nanoTime()
        val botMsg = ChatMessage(
            id = botMsgId,
            text = "",
            isUser = false,
            isStreaming = true,
            benchmark = com.example.on_device_qwen.model.BenchmarkMetrics(
                ttftMs = 0,
                decodeTps = 0.0,
                decodeTokens = 0,
                totalDurationMs = 0,
                backend = activeBackendName
            )
        )
        _messages.value = _messages.value + botMsg

        _isGenerating.value = true
        stopDolphin.set(false)

        if (dolphinEngine != null) {
            generateDolphinReply(botMsgId, startMs)
            return
        }

        viewModelScope.launch {
            try {
                val buffer = StringBuilder()
                withContext(Dispatchers.IO) {
                    if (localImageFile != null) {
                        val contents = Contents.of(
                            Content.ImageFile(localImageFile.absolutePath),
                            Content.Text(promptText)
                        )
                        conversation!!.sendMessageAsync(contents).collect { msg ->
                            val now = System.currentTimeMillis()
                            if (firstTokenMs == null) {
                                firstTokenMs = now
                            }
                            chunkCount++
                            val textPiece = msg.contents.contents
                                .filterIsInstance<Content.Text>()
                                .joinToString("") { it.text }
                            buffer.append(textPiece)
                            val currentText = buffer.toString()

                            val ttft = (firstTokenMs!! - startMs).coerceAtLeast(0)
                            val decodeDurationSec = (now - firstTokenMs!!) / 1000.0
                            val liveTps = if (decodeDurationSec > 0.05) chunkCount / decodeDurationSec else 0.0

                            val liveMetrics = com.example.on_device_qwen.model.BenchmarkMetrics(
                                ttftMs = ttft,
                                decodeTps = liveTps,
                                decodeTokens = chunkCount,
                                totalDurationMs = now - startMs,
                                backend = activeBackendName
                            )

                            withContext(Dispatchers.Main) {
                                _messages.value = _messages.value.map { m ->
                                    if (m.id == botMsgId) m.copy(
                                        text = currentText,
                                        isStreaming = true,
                                        benchmark = liveMetrics
                                    ) else m
                                }
                            }
                        }
                    } else {
                        conversation!!.sendMessageAsync(promptText).collect { chunk ->
                            val now = System.currentTimeMillis()
                            if (firstTokenMs == null) {
                                firstTokenMs = now
                            }
                            chunkCount++
                            buffer.append(chunk)
                            val currentText = buffer.toString()

                            val ttft = (firstTokenMs!! - startMs).coerceAtLeast(0)
                            val decodeDurationSec = (now - firstTokenMs!!) / 1000.0
                            val liveTps = if (decodeDurationSec > 0.05) chunkCount / decodeDurationSec else 0.0

                            val liveMetrics = com.example.on_device_qwen.model.BenchmarkMetrics(
                                ttftMs = ttft,
                                decodeTps = liveTps,
                                decodeTokens = chunkCount,
                                totalDurationMs = now - startMs,
                                backend = activeBackendName
                            )

                            withContext(Dispatchers.Main) {
                                _messages.value = _messages.value.map { m ->
                                    if (m.id == botMsgId) m.copy(
                                        text = currentText,
                                        isStreaming = true,
                                        benchmark = liveMetrics
                                    ) else m
                                }
                            }
                        }
                    }
                }

                // Streaming finished: query LiteRT-LM engine for hardware-level benchmark info
                val finalTotalDurationMs = System.currentTimeMillis() - startMs
                val benchInfo = withContext(Dispatchers.IO) {
                    try {
                        conversation?.getBenchmarkInfo()
                    } catch (_: Exception) {
                        null
                    }
                }

                val finalMetrics = if (benchInfo != null && benchInfo.lastDecodeTokensPerSecond > 0) {
                    val ttft = (benchInfo.timeToFirstTokenInSecond * 1000).toLong().coerceAtLeast(0)
                    com.example.on_device_qwen.model.BenchmarkMetrics(
                        ttftMs = if (ttft > 0) ttft else ((firstTokenMs ?: System.currentTimeMillis()) - startMs).coerceAtLeast(0),
                        decodeTps = benchInfo.lastDecodeTokensPerSecond,
                        prefillTps = if (benchInfo.lastPrefillTokensPerSecond > 0) benchInfo.lastPrefillTokensPerSecond else null,
                        prefillTokens = if (benchInfo.lastPrefillTokenCount > 0) benchInfo.lastPrefillTokenCount else null,
                        decodeTokens = if (benchInfo.lastDecodeTokenCount > 0) benchInfo.lastDecodeTokenCount else chunkCount,
                        totalDurationMs = finalTotalDurationMs,
                        backend = activeBackendName
                    )
                } else {
                    val ttft = ((firstTokenMs ?: System.currentTimeMillis()) - startMs).coerceAtLeast(0)
                    val decodeDurationSec = (finalTotalDurationMs - ttft) / 1000.0
                    val fallbackTps = if (decodeDurationSec > 0.05) chunkCount / decodeDurationSec else 0.0
                    com.example.on_device_qwen.model.BenchmarkMetrics(
                        ttftMs = ttft,
                        decodeTps = fallbackTps,
                        decodeTokens = chunkCount,
                        totalDurationMs = finalTotalDurationMs,
                        backend = activeBackendName
                    )
                }

                // Mark streaming as complete with final benchmark metrics
                _messages.value = _messages.value.map { msg ->
                    if (msg.id == botMsgId) msg.copy(
                        isStreaming = false,
                        benchmark = finalMetrics
                    ) else msg
                }
            } catch (e: Exception) {
                _messages.value = _messages.value.map { msg ->
                    if (msg.id == botMsgId) msg.copy(
                        text = "⚠️ Error: ${e.localizedMessage ?: e.toString()}",
                        isStreaming = false
                    ) else msg
                }
            } finally {
                _isGenerating.value = false
            }
        }
    }

    /** Formats Dolphin's native ChatML template using recent local conversation history. */
    private fun buildDolphinPrompt(): String = buildString {
        append("<|im_start|>system\n")
        append("You are Dolphin, a capable offline AI assistant. Be accurate, helpful, and direct.")
        append("<|im_end|>\n")
        _messages.value.filter { it.text.isNotBlank() && (it.isUser || !it.isStreaming) }
            .takeLast(12)
            .forEach { msg ->
                append(if (msg.isUser) "<|im_start|>user\n" else "<|im_start|>assistant\n")
                append(msg.text)
                append("<|im_end|>\n")
            }
        append("<|im_start|>assistant\n")
    }

    private fun generateDolphinReply(botMsgId: Long, startMs: Long) {
        val active = dolphinEngine ?: return
        val prompt = buildDolphinPrompt()
        viewModelScope.launch {
            val buffer = ByteArrayOutputStream()
            val began = SystemClock.elapsedRealtime()
            var lastUiUpdate = began
            var firstPiece = 0L
            var pieces = 0
            try {
                val tokens = withContext(Dispatchers.IO) {
                    active.generate(prompt, ChunkListener { part ->
                        val now = SystemClock.elapsedRealtime()
                        if (firstPiece == 0L) firstPiece = now
                        buffer.write(part)
                        pieces++
                        // Throttled Flow writes keep the Compose chat list responsive.
                        if (now - lastUiUpdate >= 80L) {
                            lastUiUpdate = now
                            val preview = buffer.toString("UTF-8")
                            val metrics = BenchmarkMetrics(
                                ttftMs = firstPiece - began,
                                decodeTps = if (now > firstPiece) pieces * 1000.0 / (now - firstPiece) else 0.0,
                                decodeTokens = pieces,
                                totalDurationMs = System.currentTimeMillis() - startMs,
                                backend = "CPU · llama.cpp (estimated)"
                            )
                            _messages.update { list -> list.map { msg ->
                                if (msg.id == botMsgId) msg.copy(text = preview, benchmark = metrics) else msg
                            } }
                        }
                        !stopDolphin.get()
                    })
                }
                val elapsed = SystemClock.elapsedRealtime() - began
                val firstMs = if (firstPiece == 0L) elapsed else firstPiece - began
                val metrics = BenchmarkMetrics(
                    ttftMs = firstMs,
                    decodeTps = if (elapsed > firstMs) tokens * 1000.0 / (elapsed - firstMs) else 0.0,
                    decodeTokens = tokens.coerceAtLeast(0),
                    totalDurationMs = System.currentTimeMillis() - startMs,
                    backend = "CPU · llama.cpp (estimated)"
                )
                _messages.update { list -> list.map { msg ->
                    if (msg.id == botMsgId) msg.copy(
                        text = buffer.toString("UTF-8"), isStreaming = false, benchmark = metrics
                    ) else msg
                } }
            } catch (e: Exception) {
                _messages.update { list -> list.map { msg ->
                    if (msg.id == botMsgId) msg.copy(
                        text = "⚠️ GGUF inference failed: ${e.localizedMessage}", isStreaming = false
                    ) else msg
                } }
            } finally {
                _isGenerating.value = false
                if (deferredNativeClose.getAndSet(false)) {
                    withContext(NonCancellable + Dispatchers.IO) { active.close() }
                }
            }
        }
    }

    fun stopGeneration() {
        if (_isGgufModel.value) stopDolphin.set(true)
    }

    /**
     * Clear conversation history and create a fresh session.
     */
    fun clearChat() {
        if (_isGenerating.value) return
        if (_isGgufModel.value) {
            _messages.value = emptyList()
            return
        }
        viewModelScope.launch {
            _messages.value = emptyList()
            try {
                withContext(Dispatchers.IO) {
                    conversation?.close()
                    val convConfig = ConversationConfig(
                        systemInstruction = Contents.of(SYSTEM_PROMPT)
                    )
                    conversation = engine?.createConversation(convConfig)
                }
            } catch (_: Exception) {
                // Silently ignore cleanup errors
            }
        }
    }

    override fun onCleared() {
        stopDolphin.set(true)
        if (_isGenerating.value && _isGgufModel.value) {
            deferredNativeClose.set(true)
        } else {
            dolphinEngine?.close()
        }
        super.onCleared()
        try {
            conversation?.close()
            engine?.close()
        } catch (_: Exception) {
            // Best-effort cleanup
        }
    }

    companion object {
        /** Expected model filename — change this if you use a different variant. */
        const val MODEL_FILENAME = "Qwen2.5-3B-Instruct.litertlm"

        const val SYSTEM_PROMPT =
            "You are Qwen, a helpful and knowledgeable AI assistant created by Alibaba Cloud. " +
            "You are running locally on this Android device using the LiteRT-LM engine. " +
            "Be concise, accurate, and helpful. If you are unsure about something, say so."
    }
}
