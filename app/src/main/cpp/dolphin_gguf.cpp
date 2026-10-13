// Minimal, CPU-only llama.cpp bridge for Dolphin 3.0 Qwen2.5 3B GGUF.
// Intentionally avoids retaining a Java reference or doing work on Android's UI thread.
#include <jni.h>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "llama.h"

namespace {
struct Session {
    llama_model* model = nullptr;
    llama_context* context = nullptr;
};

void throw_java(JNIEnv* env, const char* message) {
    jclass cls = env->FindClass("java/lang/IllegalStateException");
    if (cls != nullptr) env->ThrowNew(cls, message);
}

jlong native_open(JNIEnv* env, jobject, jstring path, jint context_size, jint threads) {
    if (path == nullptr) { throw_java(env, "Missing model file"); return 0; }
    static std::once_flag init_flag;
    std::call_once(init_flag, [] { llama_backend_init(); });
    const char* utf_path = env->GetStringUTFChars(path, nullptr);
    if (!utf_path) return 0;
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU-only: do not pretend GPU/NPU is supported.
    model_params.use_mmap = true;
    model_params.use_mlock = false;
    llama_model* model = llama_load_model_from_file(utf_path, model_params);
    env->ReleaseStringUTFChars(path, utf_path);
    if (!model) { throw_java(env, "Unable to load GGUF weights (format or memory error)"); return 0; }
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = (uint32_t) std::clamp((int) context_size, 1024, 4096);
    ctx_params.n_batch = 128;
    ctx_params.n_ubatch = 128;
    ctx_params.n_threads = std::clamp((int) threads, 1, 4);
    ctx_params.n_threads_batch = ctx_params.n_threads;
    llama_context* context = llama_new_context_with_model(model, ctx_params);
    if (!context) {
        llama_free_model(model);
        throw_java(env, "Unable to allocate model context; try a smaller quantization");
        return 0;
    }
    return reinterpret_cast<jlong>(new Session{model, context});
}

void native_close(JNIEnv*, jobject, jlong handle) {
    Session* session = reinterpret_cast<Session*>(handle);
    if (!session) return;
    if (session->context) llama_free(session->context);
    if (session->model) llama_free_model(session->model);
    delete session;
}

jint native_generate(JNIEnv* env, jobject, jlong handle, jstring prompt, jint max_new_tokens, jobject listener) {
    Session* s = reinterpret_cast<Session*>(handle);
    if (!s || !prompt || !listener) { throw_java(env, "No GGUF model loaded"); return -1; }
    jclass listener_class = env->GetObjectClass(listener);
    jmethodID emit = env->GetMethodID(listener_class, "onBytes", "([B)Z");
    env->DeleteLocalRef(listener_class);
    if (!emit) return -1;

    const char* input = env->GetStringUTFChars(prompt, nullptr);
    if (!input) return -1;
    std::string text(input);
    env->ReleaseStringUTFChars(prompt, input);

    const int n_ctx = (int) llama_n_ctx(s->context);
    const int max_output = std::clamp((int) max_new_tokens, 1, 512);
    // Prompt prefill must leave space for new tokens in the context window.
    const int capacity = n_ctx - max_output - 8;
    if (capacity <= 0) { throw_java(env, "Context window too small"); return -1; }

    std::vector<llama_token> tokens(std::max(128, (int) text.size() / 2 + 16));
    int count = llama_tokenize(s->model, text.c_str(), (int) text.size(),
                               tokens.data(), (int) tokens.size(), false, true);
    if (count < 0) {
        tokens.resize(-count);
        count = llama_tokenize(s->model, text.c_str(), (int) text.size(),
                               tokens.data(), (int) tokens.size(), false, true);
    }
    if (count <= 0 || count > capacity) {
        throw_java(env, "Conversation exceeds the GGUF context window. Start a new chat.");
        return -1;
    }
    tokens.resize(count);
    llama_kv_cache_clear(s->context);
    for (int pos = 0; pos < count; pos += 128) {
        int n = std::min(128, count - pos);
        llama_batch batch = llama_batch_get_one(tokens.data() + pos, n);
        if (llama_decode(s->context, batch) != 0) {
            throw_java(env, "GGUF prompt processing failed");
            return -1;
        }
    }

    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    int generated = 0;
    for (; generated < max_output; ++generated) {
        llama_token token = llama_sampler_sample(sampler, s->context, -1);
        llama_sampler_accept(sampler, token);
        if (llama_token_is_eog(s->model, token)) break;

        char stack_buffer[256];
        int size = llama_token_to_piece(s->model, token, stack_buffer, sizeof(stack_buffer), 0, false);
        std::vector<char> extra;
        const char* piece = stack_buffer;
        if (size < 0) {
            extra.resize((size_t) -size);
            size = llama_token_to_piece(s->model, token, extra.data(), (int) extra.size(), 0, false);
            piece = extra.data();
        }
        if (size > 0) {
            jbyteArray bytes = env->NewByteArray(size);
            if (!bytes) break;
            env->SetByteArrayRegion(bytes, 0, size, reinterpret_cast<const jbyte*>(piece));
            jboolean keep_going = env->CallBooleanMethod(listener, emit, bytes);
            env->DeleteLocalRef(bytes);
            if (env->ExceptionCheck() || keep_going != JNI_TRUE) break;
        }
        llama_batch one = llama_batch_get_one(&token, 1);
        if (llama_decode(s->context, one) != 0) {
            llama_sampler_free(sampler);
            throw_java(env, "GGUF decoding failed");
            return -1;
        }
    }
    llama_sampler_free(sampler);
    return generated;
}
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;
    jclass cls = env->FindClass("com/example/on_device_qwen/inference/DolphinNativeEngine");
    if (!cls) return JNI_ERR;
    static JNINativeMethod methods[] = {
        {"nativeOpen", "(Ljava/lang/String;II)J", reinterpret_cast<void*>(native_open)},
        {"nativeGenerate", "(JLjava/lang/String;ILcom/example/on_device_qwen/inference/ChunkListener;)I", reinterpret_cast<void*>(native_generate)},
        {"nativeClose", "(J)V", reinterpret_cast<void*>(native_close)}
    };
    int result = env->RegisterNatives(cls, methods, 3);
    env->DeleteLocalRef(cls);
    return result == 0 ? JNI_VERSION_1_6 : JNI_ERR;
}
