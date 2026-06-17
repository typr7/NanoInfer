#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cuda_device_buffer.h"
#include "llama.h"
#include "macro.h"
#include "kernels/attention_softmax.h"
#include "kernels/greedy_decoding.h"
#include "kernels/grouped_query_attention.h"
#include "kernels/kv_cache.h"
#include "kernels/residual_connection.h"
#include "kernels/rms_norm.h"
#include "kernels/rope.h"
#include "kernels/swiglu.h"
#include "kernels/token_embedding.h"


namespace
{

__nv_bfloat16 bf16(float value)
{
    return __float2bfloat16(value);
}

float f32(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

void sync_and_check()
{
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void expect_near(float actual, float expected, float tolerance, const std::string& context)
{
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            context + ": actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected)
        );
    }
}

void expect_equal(int actual, int expected, const std::string& context)
{
    if (actual != expected) {
        throw std::runtime_error(
            context + ": actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected)
        );
    }
}

void upload(CudaDeviceBuffer& buffer, const std::vector<__nv_bfloat16>& values)
{
    buffer.upload(values.data(), values.size() * sizeof(__nv_bfloat16));
}

std::vector<__nv_bfloat16> download_bf16(const CudaDeviceBuffer& buffer, std::size_t count)
{
    std::vector<__nv_bfloat16> values(count);
    buffer.download(values.data(), values.size() * sizeof(__nv_bfloat16));
    return values;
}

float llama3_inv_freq(int pair_idx)
{
    const float freq = std::pow(
        ROPE_THETA,
        static_cast<float>(2 * pair_idx) / static_cast<float>(HEAD_DIM)
    );
    const float wavelen = 2.f * static_cast<float>(M_PI) * freq;
    const float low_freq_wavelen =
        ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS / ROPE_SCALING_LOW_FREQ_FACTOR;
    const float high_freq_wavelen =
        ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS / ROPE_SCALING_HIGH_FREQ_FACTOR;

    float inv_freq = 1.f / freq;
    if (wavelen > low_freq_wavelen) {
        inv_freq /= ROPE_SCALING_FACTOR;
    } else if (wavelen >= high_freq_wavelen) {
        const float smooth =
            (ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS / wavelen -
             ROPE_SCALING_LOW_FREQ_FACTOR) /
            (ROPE_SCALING_HIGH_FREQ_FACTOR - ROPE_SCALING_LOW_FREQ_FACTOR);
        inv_freq = (1.f - smooth) * inv_freq / ROPE_SCALING_FACTOR + smooth * inv_freq;
    }

    return inv_freq;
}

void apply_rope_reference_to_projection(
    int position,
    int projection_dim,
    __nv_bfloat16* projection
) {
    constexpr int half_head_dim = HEAD_DIM / 2;
    for (int i = 0; i < projection_dim / 2; i++) {
        const int pair_idx = i % half_head_dim;
        const float freq = llama3_inv_freq(pair_idx);
        const float sin_theta = std::sin(position * freq);
        const float cos_theta = std::cos(position * freq);

        const int idx = 2 * i - pair_idx;
        const float val1 = f32(projection[idx]);
        const float val2 = f32(projection[idx + half_head_dim]);
        projection[idx] = bf16(val1 * cos_theta - val2 * sin_theta);
        projection[idx + half_head_dim] = bf16(val1 * sin_theta + val2 * cos_theta);
    }
}

void test_token_embedding()
{
    constexpr std::size_t token_count = 3;
    constexpr int embedding_rows = 5;
    const std::vector<std::int32_t> token_ids = {2, 0, 4};

    std::vector<__nv_bfloat16> table(embedding_rows * HIDDEN_DIM);
    for (int row = 0; row < embedding_rows; row++) {
        for (int dim = 0; dim < HIDDEN_DIM; dim++) {
            table[row * HIDDEN_DIM + dim] = bf16(0.125f * row + 0.001f * (dim % 97));
        }
    }

    CudaDeviceBuffer token_id_buffer(token_ids.size() * sizeof(std::int32_t));
    CudaDeviceBuffer table_buffer(table.size() * sizeof(__nv_bfloat16));
    CudaDeviceBuffer output_buffer(token_count * HIDDEN_DIM * sizeof(__nv_bfloat16));
    token_id_buffer.upload(token_ids.data(), token_ids.size() * sizeof(std::int32_t));
    upload(table_buffer, table);

    launch_token_embedding_kernel(
        token_count,
        token_id_buffer.data<std::int32_t>(),
        table_buffer.data<__nv_bfloat16>(),
        output_buffer.data<__nv_bfloat16>(),
        nullptr
    );
    sync_and_check();

    const std::vector<__nv_bfloat16> output = download_bf16(output_buffer, token_count * HIDDEN_DIM);
    for (std::size_t token = 0; token < token_count; token++) {
        const int source_row = token_ids[token];
        for (int dim = 0; dim < HIDDEN_DIM; dim++) {
            const float actual = f32(output[token * HIDDEN_DIM + dim]);
            const float expected = f32(table[source_row * HIDDEN_DIM + dim]);
            expect_near(actual, expected, 0.f, "token_embedding");
        }
    }
}

void test_greedy_decoding()
{
    constexpr int selected_token_id = 7;
    constexpr int tied_larger_token_id = 11;
    constexpr int embedding_rows = 16;

    std::vector<__nv_bfloat16> logits(VOCAB_LEN, bf16(-10.f));
    logits[selected_token_id] = bf16(3.5f);
    logits[tied_larger_token_id] = bf16(3.5f);
    logits[3] = bf16(2.75f);

    std::vector<__nv_bfloat16> embedding_table(embedding_rows * HIDDEN_DIM);
    for (int row = 0; row < embedding_rows; row++) {
        for (int dim = 0; dim < HIDDEN_DIM; dim++) {
            embedding_table[row * HIDDEN_DIM + dim] =
                bf16(0.03125f * row + 0.0005f * static_cast<float>(dim % 53));
        }
    }

    CudaDeviceBuffer logits_buffer(logits.size() * sizeof(__nv_bfloat16));
    CudaDeviceBuffer embedding_buffer(embedding_table.size() * sizeof(__nv_bfloat16));
    CudaDeviceBuffer next_embedding_buffer(HIDDEN_DIM * sizeof(__nv_bfloat16));
    CudaDeviceBuffer next_token_id_buffer(sizeof(std::int32_t));
    upload(logits_buffer, logits);
    upload(embedding_buffer, embedding_table);

    launch_single_batch_greedy_decode_kernel(
        logits_buffer.data<__nv_bfloat16>(),
        embedding_buffer.data<__nv_bfloat16>(),
        next_embedding_buffer.data<__nv_bfloat16>(),
        next_token_id_buffer.data<std::int32_t>(),
        nullptr
    );
    sync_and_check();

    std::int32_t next_token_id = -1;
    next_token_id_buffer.download(&next_token_id, sizeof(next_token_id));
    expect_equal(next_token_id, selected_token_id, "greedy_decoding token id");

    const std::vector<__nv_bfloat16> output = download_bf16(next_embedding_buffer, HIDDEN_DIM);
    for (int dim = 0; dim < HIDDEN_DIM; dim++) {
        const float expected = f32(embedding_table[selected_token_id * HIDDEN_DIM + dim]);
        expect_near(f32(output[dim]), expected, 0.f, "greedy_decoding embedding");
    }
}

void test_residual_connection()
{
    constexpr std::size_t token_count = 2;
    const std::size_t count = token_count * HIDDEN_DIM;

    std::vector<__nv_bfloat16> residual(count);
    std::vector<__nv_bfloat16> hidden(count);
    for (std::size_t i = 0; i < count; i++) {
        residual[i] = bf16(static_cast<float>(static_cast<int>(i % 19) - 9) * 0.0625f);
        hidden[i] = bf16(static_cast<float>(static_cast<int>(i % 23) - 11) * 0.125f);
    }

    CudaDeviceBuffer residual_buffer(count * sizeof(__nv_bfloat16));
    CudaDeviceBuffer hidden_buffer(count * sizeof(__nv_bfloat16));
    upload(residual_buffer, residual);
    upload(hidden_buffer, hidden);

    launch_residual_connection_kernel(
        token_count,
        residual_buffer.data<__nv_bfloat16>(),
        hidden_buffer.data<__nv_bfloat16>(),
        nullptr
    );
    sync_and_check();

    const std::vector<__nv_bfloat16> output = download_bf16(hidden_buffer, count);
    for (std::size_t i = 0; i < count; i++) {
        const float expected = f32(hidden[i]) + f32(residual[i]);
        expect_near(f32(output[i]), expected, 0.03125f, "residual_connection");
    }
}

void test_rms_norm()
{
    constexpr std::size_t token_count = 2;
    const std::size_t count = token_count * HIDDEN_DIM;

    std::vector<__nv_bfloat16> hidden(count);
    std::vector<__nv_bfloat16> norm_weights(HIDDEN_DIM);
    for (std::size_t i = 0; i < count; i++) {
        hidden[i] = bf16(static_cast<float>(static_cast<int>(i % 37) - 18) * 0.0625f);
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        norm_weights[i] = bf16(0.75f + static_cast<float>(i % 17) * 0.0078125f);
    }

    CudaDeviceBuffer hidden_buffer(count * sizeof(__nv_bfloat16));
    CudaDeviceBuffer norm_buffer(norm_weights.size() * sizeof(__nv_bfloat16));
    CudaDeviceBuffer output_buffer(count * sizeof(__nv_bfloat16));
    upload(hidden_buffer, hidden);
    upload(norm_buffer, norm_weights);

    launch_rms_norm_kernel(
        token_count,
        hidden_buffer.data<__nv_bfloat16>(),
        norm_buffer.data<__nv_bfloat16>(),
        output_buffer.data<__nv_bfloat16>(),
        nullptr
    );
    sync_and_check();

    const std::vector<__nv_bfloat16> output = download_bf16(output_buffer, count);
    for (std::size_t token = 0; token < token_count; token++) {
        float sum_squares = 0.f;
        for (int dim = 0; dim < HIDDEN_DIM; dim++) {
            const float value = f32(hidden[token * HIDDEN_DIM + dim]);
            sum_squares += value * value;
        }
        const float rms_rfactor = rsqrtf(sum_squares / HIDDEN_DIM + RMS_NORM_EPS);
        for (int dim = 0; dim < HIDDEN_DIM; dim++) {
            const float expected = f32(hidden[token * HIDDEN_DIM + dim]) *
                                   f32(norm_weights[dim]) *
                                   rms_rfactor;
            expect_near(
                f32(output[token * HIDDEN_DIM + dim]),
                expected,
                0.02f,
                "rms_norm"
            );
        }
    }
}

void test_swiglu()
{
    constexpr std::size_t token_count = 3;
    constexpr int activation_dim = 128;
    constexpr int gate_up_stride = 2 * activation_dim;
    const std::size_t count = token_count * gate_up_stride;

    std::vector<__nv_bfloat16> gate_up(count);
    for (std::size_t i = 0; i < count; i++) {
        gate_up[i] = bf16(static_cast<float>(static_cast<int>(i % 29) - 14) * 0.03125f);
    }
    const std::vector<__nv_bfloat16> original = gate_up;

    CudaDeviceBuffer gate_up_buffer(count * sizeof(__nv_bfloat16));
    upload(gate_up_buffer, gate_up);

    launch_swiglu_inplace_kernel(
        token_count,
        activation_dim,
        gate_up_stride,
        gate_up_buffer.data<__nv_bfloat16>(),
        nullptr
    );
    sync_and_check();

    const std::vector<__nv_bfloat16> output = download_bf16(gate_up_buffer, count);
    for (std::size_t token = 0; token < token_count; token++) {
        const std::size_t base = token * gate_up_stride;
        for (int dim = 0; dim < activation_dim; dim++) {
            const float gate = f32(original[base + dim]);
            const float up = f32(original[base + activation_dim + dim]);
            const float expected = up * gate / (1.f + std::exp(-gate));
            expect_near(f32(output[base + dim]), expected, 0.01f, "swiglu gate");
            expect_near(
                f32(output[base + activation_dim + dim]),
                f32(original[base + activation_dim + dim]),
                0.f,
                "swiglu up"
            );
        }
    }
}

void test_attention_softmax()
{
    constexpr std::size_t head_count = 2;
    constexpr std::size_t token_count = 4;
    const std::size_t count = head_count * token_count * token_count;

    std::vector<__nv_bfloat16> scores(count);
    for (std::size_t head = 0; head < head_count; head++) {
        for (std::size_t row = 0; row < token_count; row++) {
            for (std::size_t col = 0; col < token_count; col++) {
                const std::size_t idx = head * token_count * token_count +
                                        row * token_count + col;
                scores[idx] = bf16(0.1f * static_cast<float>(head + 1) +
                                   0.2f * static_cast<float>(row) -
                                   0.15f * static_cast<float>(col));
            }
        }
    }

    CudaDeviceBuffer scores_buffer(count * sizeof(__nv_bfloat16));
    upload(scores_buffer, scores);

    launch_attention_softmax_prefill_kernel(
        head_count,
        token_count,
        scores_buffer.data<__nv_bfloat16>(),
        nullptr
    );
    sync_and_check();

    const std::vector<__nv_bfloat16> output = download_bf16(scores_buffer, count);
    for (std::size_t head = 0; head < head_count; head++) {
        for (std::size_t row = 0; row < token_count; row++) {
            float max_value = -INFINITY;
            for (std::size_t col = 0; col <= row; col++) {
                const std::size_t idx = head * token_count * token_count +
                                        row * token_count + col;
                max_value = std::fmax(max_value, f32(scores[idx]));
            }
            float sum = 0.f;
            for (std::size_t col = 0; col <= row; col++) {
                const std::size_t idx = head * token_count * token_count +
                                        row * token_count + col;
                sum += std::exp(f32(scores[idx]) - max_value);
            }
            for (std::size_t col = 0; col < token_count; col++) {
                const std::size_t idx = head * token_count * token_count +
                                        row * token_count + col;
                const float expected = (col <= row)
                    ? std::exp(f32(scores[idx]) - max_value) / sum
                    : 0.f;
                expect_near(f32(output[idx]), expected, 0.01f, "attention_softmax");
            }
        }
    }
}

void test_kv_cache_store()
{
    constexpr std::size_t lines_to_store = 3;
    constexpr int layer_idx = 0;
    constexpr int line_offset = 2;
    constexpr int cache_stride = 2 * K_PROJ_DIM;
    constexpr int kv_stride = cache_stride + 16;
    constexpr std::size_t cache_lines = line_offset + lines_to_store + 1;

    std::vector<__nv_bfloat16> kv(lines_to_store * kv_stride);
    std::vector<__nv_bfloat16> cache(cache_lines * cache_stride, bf16(-7.f));
    for (std::size_t line = 0; line < lines_to_store; line++) {
        for (int dim = 0; dim < kv_stride; dim++) {
            kv[line * kv_stride + dim] = bf16(0.01f * static_cast<float>(line * kv_stride + dim));
        }
    }

    CudaDeviceBuffer kv_buffer(kv.size() * sizeof(__nv_bfloat16));
    CudaDeviceBuffer cache_buffer(cache.size() * sizeof(__nv_bfloat16));
    upload(kv_buffer, kv);
    upload(cache_buffer, cache);

    launch_single_batch_store_kv_cache_kernel(
        lines_to_store,
        layer_idx,
        line_offset,
        kv_stride,
        kv_buffer.data<__nv_bfloat16>(),
        cache_buffer.data<__nv_bfloat16>(),
        nullptr
    );
    sync_and_check();

    const std::vector<__nv_bfloat16> output = download_bf16(cache_buffer, cache.size());
    for (std::size_t line = 0; line < cache_lines; line++) {
        for (int dim = 0; dim < cache_stride; dim++) {
            const bool written = line >= line_offset && line < line_offset + lines_to_store;
            const float expected = written
                ? f32(kv[(line - line_offset) * kv_stride + dim])
                : f32(cache[line * cache_stride + dim]);
            expect_near(f32(output[line * cache_stride + dim]), expected, 0.f, "kv_cache");
        }
    }
}

void test_grouped_query_attention_decode()
{
    constexpr std::size_t token_count = 5;
    constexpr int layer_idx = 0;
    constexpr int group_size = Q_HEAD_NUM / K_HEAD_NUM;
    constexpr int cache_stride = 2 * K_PROJ_DIM;
    constexpr float rsqrt_head_dim = 1.f / 8.f;

    std::vector<__nv_bfloat16> q(Q_PROJ_DIM);
    std::vector<__nv_bfloat16> kv_cache(token_count * cache_stride);
    for (int i = 0; i < Q_PROJ_DIM; i++) {
        q[i] = bf16(static_cast<float>(static_cast<int>(i % 23) - 11) * 0.015625f);
    }
    for (std::size_t token = 0; token < token_count; token++) {
        for (int dim = 0; dim < cache_stride; dim++) {
            kv_cache[token * cache_stride + dim] =
                bf16(static_cast<float>(static_cast<int>((token * 17 + dim) % 29) - 14) * 0.01f);
        }
    }

    CudaDeviceBuffer q_buffer(q.size() * sizeof(__nv_bfloat16));
    CudaDeviceBuffer kv_cache_buffer(kv_cache.size() * sizeof(__nv_bfloat16));
    CudaDeviceBuffer output_buffer(Q_PROJ_DIM * sizeof(__nv_bfloat16));
    upload(q_buffer, q);
    upload(kv_cache_buffer, kv_cache);

    launch_single_batch_gqa_decode_kernel(
        token_count,
        layer_idx,
        q_buffer.data<__nv_bfloat16>(),
        kv_cache_buffer.data<__nv_bfloat16>(),
        output_buffer.data<__nv_bfloat16>(),
        nullptr
    );
    sync_and_check();

    const std::vector<__nv_bfloat16> output = download_bf16(output_buffer, Q_PROJ_DIM);
    for (int q_head_idx = 0; q_head_idx < Q_HEAD_NUM; q_head_idx++) {
        const int k_head_idx = q_head_idx / group_size;
        const __nv_bfloat16* query_head = q.data() + q_head_idx * HEAD_DIM;
        const __nv_bfloat16* key_head = kv_cache.data() + k_head_idx * HEAD_DIM;
        const __nv_bfloat16* value_head = key_head + K_PROJ_DIM;

        std::vector<float> scores(token_count, 0.f);
        float row_max = -INFINITY;
        for (std::size_t token = 0; token < token_count; token++) {
            float score = 0.f;
            for (int dim = 0; dim < HEAD_DIM; dim++) {
                score += f32(query_head[dim]) *
                         f32(key_head[token * cache_stride + dim]);
            }
            scores[token] = score * rsqrt_head_dim;
            row_max = std::fmax(row_max, scores[token]);
        }

        float score_sum = 0.f;
        for (float& score: scores) {
            score = std::exp(score - row_max);
            score_sum += score;
        }

        for (int dim = 0; dim < HEAD_DIM; dim++) {
            float expected = 0.f;
            for (std::size_t token = 0; token < token_count; token++) {
                const float probability = scores[token] / score_sum;
                expected += probability * f32(value_head[token * cache_stride + dim]);
            }
            expect_near(
                f32(output[q_head_idx * HEAD_DIM + dim]),
                expected,
                0.015f,
                "grouped_query_attention"
            );
        }
    }
}

void test_rope_position_zero_identity()
{
    constexpr std::size_t token_count = 1;
    constexpr int qk_stride = Q_PROJ_DIM + 2 * K_PROJ_DIM;
    std::vector<__nv_bfloat16> qkv(token_count * qk_stride);
    for (std::size_t i = 0; i < qkv.size(); i++) {
        qkv[i] = bf16(static_cast<float>(static_cast<int>(i % 31) - 15) * 0.015625f);
    }
    const std::vector<__nv_bfloat16> original = qkv;

    CudaDeviceBuffer qkv_buffer(qkv.size() * sizeof(__nv_bfloat16));
    upload(qkv_buffer, qkv);

    launch_rope_llama3_half_split_inplace_kernel(
        token_count,
        0,
        qk_stride,
        qkv_buffer.data<__nv_bfloat16>(),
        nullptr
    );
    sync_and_check();

    const std::vector<__nv_bfloat16> output = download_bf16(qkv_buffer, qkv.size());
    for (std::size_t i = 0; i < qkv.size(); i++) {
        expect_near(f32(output[i]), f32(original[i]), 0.f, "rope position zero");
    }
}

void test_rope_nonzero_positions()
{
    constexpr std::size_t token_count = 3;
    constexpr int position_offset = 5;
    constexpr int qk_stride = Q_PROJ_DIM + 2 * K_PROJ_DIM;

    std::vector<__nv_bfloat16> qkv(token_count * qk_stride);
    for (std::size_t i = 0; i < qkv.size(); i++) {
        qkv[i] = bf16(static_cast<float>(static_cast<int>(i % 43) - 21) * 0.01171875f);
    }
    std::vector<__nv_bfloat16> expected = qkv;

    for (std::size_t token = 0; token < token_count; token++) {
        const int position = position_offset + static_cast<int>(token);
        __nv_bfloat16* token_begin = expected.data() + token * qk_stride;
        apply_rope_reference_to_projection(position, Q_PROJ_DIM, token_begin);
        apply_rope_reference_to_projection(position, K_PROJ_DIM, token_begin + Q_PROJ_DIM);
    }

    CudaDeviceBuffer qkv_buffer(qkv.size() * sizeof(__nv_bfloat16));
    upload(qkv_buffer, qkv);

    launch_rope_llama3_half_split_inplace_kernel(
        token_count,
        position_offset,
        qk_stride,
        qkv_buffer.data<__nv_bfloat16>(),
        nullptr
    );
    sync_and_check();

    const std::vector<__nv_bfloat16> output = download_bf16(qkv_buffer, qkv.size());
    for (std::size_t token = 0; token < token_count; token++) {
        const std::size_t base = token * qk_stride;
        for (int dim = 0; dim < qk_stride; dim++) {
            const float tolerance = dim < Q_PROJ_DIM + K_PROJ_DIM ? 0.01f : 0.f;
            expect_near(
                f32(output[base + dim]),
                f32(expected[base + dim]),
                tolerance,
                "rope nonzero positions"
            );
        }
    }
}

bool has_cuda_device()
{
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        cudaGetLastError();
        std::cout << "[SKIP] CUDA device query failed: "
                  << cudaGetErrorString(status) << '\n';
        return false;
    }
    if (device_count == 0) {
        std::cout << "[SKIP] No CUDA device found.\n";
        return false;
    }
    CUDA_CHECK(cudaSetDevice(0));
    return true;
}

} // namespace

int main()
{
    try {
        if (!has_cuda_device()) {
            return 0;
        }

        test_token_embedding();
        test_greedy_decoding();
        test_residual_connection();
        test_rms_norm();
        test_swiglu();
        test_attention_softmax();
        test_kv_cache_store();
        test_grouped_query_attention_decode();
        test_rope_position_zero_identity();
        test_rope_nonzero_positions();

        std::cout << "[PASS] kernel unit tests\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
