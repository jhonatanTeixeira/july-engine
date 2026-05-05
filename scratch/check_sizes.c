#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

enum llama_rope_scaling_type { A };
enum llama_pooling_type { B };
enum llama_attention_type { C };
enum llama_flash_attn_type { D };
enum ggml_type { E };

struct llama_sampler_seq_config;
typedef void* ggml_backend_sched_eval_callback;
typedef void* ggml_abort_callback;

struct llama_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;
    int32_t  n_threads_batch;

    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type      pooling_type;
    enum llama_attention_type    attention_type;
    enum llama_flash_attn_type   flash_attn_type;

    float    rope_freq_base;
    float    rope_freq_scale;
    float    yarn_ext_factor;
    float    yarn_attn_factor;
    float    yarn_beta_fast;
    float    yarn_beta_slow;
    uint32_t yarn_orig_ctx;
    float    defrag_thold;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    enum ggml_type type_k;
    enum ggml_type type_v;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    bool embeddings;
    bool offload_kqv;
    bool no_perf;
    bool op_offload;
    bool swa_full;
    bool kv_unified;

    struct llama_sampler_seq_config * samplers;
    size_t                            n_samplers;
};

int main() {
    printf("C++ struct size: %zu\n", sizeof(struct llama_context_params));
    printf("Offset cb_eval: %zu\n", offsetof(struct llama_context_params, cb_eval));
    printf("Offset flash_attn_type: %zu\n", offsetof(struct llama_context_params, flash_attn_type));
    printf("Offset samplers: %zu\n", offsetof(struct llama_context_params, samplers));
    printf("Offset n_samplers: %zu\n", offsetof(struct llama_context_params, n_samplers));
    return 0;
}
