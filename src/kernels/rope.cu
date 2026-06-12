#include "rope.h"
#include "../llama.h"
#include "../macro.h"

#include <math_constants.h>


namespace {

constexpr int ROPE_PAIR_COUNT = HEAD_DIM / 2;

__device__ __constant__ float d_rope_freqs[ROPE_PAIR_COUNT];

float llama3_inv_freq(int pair_idx)
{
    constexpr float rope_theta = 500000.f;
    constexpr float rope_factor = 32.f;
    constexpr float low_freq_factor = 1.f;
    constexpr float high_freq_factor = 4.f;
    constexpr float original_max_position = 8192.f;

    const float inv_freq = 1.f / powf(
        rope_theta,
        static_cast<float>(2 * pair_idx) / static_cast<float>(HEAD_DIM)
    );
    const float wavelen = 2.f * CUDART_PI_F / inv_freq;
    const float low_freq_wavelen = original_max_position / low_freq_factor;
    const float high_freq_wavelen = original_max_position / high_freq_factor;

    if (wavelen > low_freq_wavelen) {
        return inv_freq / rope_factor;
    }
    if (wavelen < high_freq_wavelen) {
        return inv_freq;
    }

    const float smooth = (original_max_position / wavelen - low_freq_factor) /
                         (high_freq_factor - low_freq_factor);
    return (1.f - smooth) * inv_freq / rope_factor + smooth * inv_freq;
}

void ensure_rope_freqs_initialized()
{
    static bool initialized = false;
    if (initialized) {
        return;
    }

    float freqs[ROPE_PAIR_COUNT];
    for (int pair_idx = 0; pair_idx < ROPE_PAIR_COUNT; pair_idx++) {
        freqs[pair_idx] = llama3_inv_freq(pair_idx);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_rope_freqs, freqs, sizeof(freqs)));
    initialized = true;
}

__global__
void rope_kernel(
    int query_stride,
    int key_stride,
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key
) {
    const int query_idx = blockIdx.x * query_stride + 2 * threadIdx.x;
    const int pair_idx = threadIdx.x % ROPE_PAIR_COUNT;
    const float theta = static_cast<float>(blockIdx.x) * d_rope_freqs[pair_idx];
    const float cos_theta = cosf(theta);
    const float sin_theta = sinf(theta);
    if (2 * threadIdx.x < Q_PROJ_DIM) {
        const float even_value = query[query_idx];
        const float odd_value = query[query_idx + 1];
        query[query_idx] = even_value * cos_theta - odd_value * sin_theta;
        query[query_idx + 1] = even_value * sin_theta + odd_value * cos_theta;
    }
    const int key_idx = blockIdx.x * key_stride + 2 * threadIdx.x;
    if (2 * threadIdx.x < K_PROJ_DIM) {
        const float even_value = key[key_idx];
        const float odd_value = key[key_idx + 1];
        key[key_idx] = even_value * cos_theta - odd_value * sin_theta;
        key[key_idx + 1] = even_value * sin_theta + odd_value * cos_theta;
    }
}

} // namespace

void launch_rope_kernel(
    std::size_t token_count,
    int query_stride,
    int key_stride,
    __nv_bfloat16* query,
    __nv_bfloat16* key,
    cudaStream_t stream
) {
    ensure_rope_freqs_initialized();

    rope_kernel<<<token_count, HIDDEN_DIM / 2, 0, stream>>>(
        query_stride,
        key_stride,
        query,
        key
    );
}
