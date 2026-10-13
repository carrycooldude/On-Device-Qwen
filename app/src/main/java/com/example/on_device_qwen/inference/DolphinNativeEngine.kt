package com.example.on_device_qwen.inference

import java.io.Closeable
import java.io.File

/** Callback receives raw UTF-8 byte fragments; a model token need not be a complete character. */
fun interface ChunkListener {
    /** Return false to stop generation at the next token boundary. */
    fun onBytes(bytes: ByteArray): Boolean
}

/**
 * Local CPU-only llama.cpp bridge for Dolphin GGUF models.
 * Do not call load(), generate() or close() on the UI thread.
 *
 * The caller owns lifecycle synchronization: generation must finish before closing the model.
 */
class DolphinNativeEngine : Closeable {
    init { System.loadLibrary("dolphin_gguf") }

    private var modelHandle: Long = 0L

    fun load(file: File) {
        require(file.isFile && file.canRead() && file.length() > 1_048_576) {
            "Readable GGUF model required"
        }
        close()
        // Reserve two CPU cores for UI/system work, cap at four to reduce thermal load.
        val processors = Runtime.getRuntime().availableProcessors()
        val threads = (processors - 2).coerceIn(1, 4)
        modelHandle = nativeOpen(file.absolutePath, 3072, threads)
        check(modelHandle != 0L) { "Unable to load GGUF model" }
    }

    fun generate(prompt: String, listener: ChunkListener): Int {
        check(modelHandle != 0L) { "GGUF model has not been loaded" }
        return nativeGenerate(modelHandle, prompt, 512, listener)
    }

    override fun close() {
        if (modelHandle != 0L) {
            nativeClose(modelHandle)
            modelHandle = 0L
        }
    }

    private external fun nativeOpen(path: String, contextSize: Int, threads: Int): Long
    private external fun nativeGenerate(
        handle: Long,
        prompt: String,
        maxNewTokens: Int,
        listener: ChunkListener
    ): Int
    private external fun nativeClose(handle: Long)
}
