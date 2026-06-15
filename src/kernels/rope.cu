#include <cassert>

#include "rope.h"
#include "../llama.h"
#include "../macro.h"

#include <math_constants.h>


namespace {

constexpr int ROPE_PAIR_COUNT = HEAD_DIM / 2;
static_assert((HEAD_DIM & 0b01) == 0);
static_assert((Q_PROJ_DIM % HEAD_DIM) == 0);
static_assert((K_PROJ_DIM % HEAD_DIM) == 0);

__device__ __constant__ float d_rope_freqs[ROPE_PAIR_COUNT];

float llama3_inv_freq(int pair_idx)
{
    constexpr float rope_theta = ROPE_THETA;
    constexpr float rope_factor = ROPE_SCALING_FACTOR;
    constexpr float low_freq_factor = ROPE_SCALING_LOW_FREQ_FACTOR;
    constexpr float high_freq_factor = ROPE_SCALING_HIGH_FREQ_FACTOR;
    constexpr float original_max_position = ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS;

    const float freq = powf(
        rope_theta,
        static_cast<float>(2 * pair_idx) / static_cast<float>(HEAD_DIM)
    );
    const float wavelen = 2.f * CUDART_PI_F * freq;
    const float low_freq_wavelen = original_max_position / low_freq_factor;
    const float high_freq_wavelen = original_max_position / high_freq_factor;

    float inv_freq = 1.f / freq;
    if (wavelen > low_freq_wavelen) {
        inv_freq /= rope_factor;
    } else if (wavelen >= high_freq_wavelen) {
        const float smooth = (original_max_position / wavelen - low_freq_factor) /
                             (high_freq_factor - low_freq_factor);
        inv_freq = (1.f - smooth) * inv_freq / rope_factor + smooth * inv_freq;
    }

    return inv_freq;
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

template <int BLOCK_SIZE> __global__
void rope_llama3_half_split_inplace_kernel(
    int position_offset,
    int qk_stride,
    __nv_bfloat16* qk
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int position = position_offset + bid;
    const int half_head_dim = (HEAD_DIM >> 1);
    __nv_bfloat16* q_begin = qk + bid * qk_stride;
    __nv_bfloat16* k_begin = q_begin + Q_PROJ_DIM;

    #pragma unroll
    for (int i = tid; i < (Q_PROJ_DIM >> 1); i += BLOCK_SIZE) {
        const int pair_idx = i % half_head_dim;
        const float freq = d_rope_freqs[pair_idx];
        const float sin_theta = sinf(position * freq);
        const float cos_theta = cosf(position * freq);

        const int idx = 2 * i - pair_idx;
        const float val1 = static_cast<float>(q_begin[idx]);
        const float val2 = static_cast<float>(q_begin[idx + half_head_dim]);
        q_begin[idx] = static_cast<__nv_bfloat16>(val1 * cos_theta - val2 * sin_theta);
        q_begin[idx + half_head_dim] = static_cast<__nv_bfloat16>(val1 * sin_theta + val2 * cos_theta);
    }

    #pragma unroll
    for (int i = tid; i < (K_PROJ_DIM >> 1); i += BLOCK_SIZE) {
        const int pair_idx = i % half_head_dim;
        const float freq = d_rope_freqs[pair_idx];
        const float sin_theta = sinf(position * freq);
        const float cos_theta = cosf(position * freq);

        const int idx = 2 * i - pair_idx;
        const float val1 = static_cast<float>(k_begin[idx]);
        const float val2 = static_cast<float>(k_begin[idx + half_head_dim]);
        k_begin[idx] = static_cast<__nv_bfloat16>(val1 * cos_theta - val2 * sin_theta);
        k_begin[idx + half_head_dim] = static_cast<__nv_bfloat16>(val1 * sin_theta + val2 * cos_theta);
    }
}

} // namespace

void launch_rope_llama3_half_split_inplace_kernel(
    std::size_t token_count,
    int position_offset,
    int qk_stride,
    __nv_bfloat16* qk,
    cudaStream_t stream
) {
    ensure_rope_freqs_initialized();

    constexpr int BLOCK_SIZE = 256;
    rope_llama3_half_split_inplace_kernel<BLOCK_SIZE>
        <<<token_count, BLOCK_SIZE, 0, stream>>>(position_offset, qk_stride, qk);
}
