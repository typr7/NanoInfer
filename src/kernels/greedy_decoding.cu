#include <math_constants.h>

#include "greedy_decoding.h"
#include "../llama.h"


namespace
{

struct Pair
{
    float val;
    int idx;
};

__device__ __forceinline__
bool better(Pair p1, Pair p2)
{
    return (p1.val > p2.val) || (p1.val == p2.val && p1.idx < p2.idx);
}


template<int BLOCK_SIZE> __global__
void single_batch_greedy_decode_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const __nv_bfloat16* __restrict__ embedding_table,
    __nv_bfloat16* __restrict__ next_token_embedding,
    int32_t* __restrict__ next_token_id
) {
    __shared__ float val[BLOCK_SIZE];
    __shared__ int idx[BLOCK_SIZE];

    const int tid = threadIdx.x;

    Pair best{ -CUDART_INF_F, -1 };

    for (int i = tid; i < VOCAB_LEN; i += BLOCK_SIZE) {
        auto cand = Pair{ static_cast<float>(logits[i]), i };
        if (better(cand, best)) {
            best = cand;
        }
    }

    val[tid] = best.val;
    idx[tid] = best.idx;

    __syncthreads();

    if (tid == 0) {
        best.val = val[0];
        best.idx = idx[0];

        for (int i = 1; i < BLOCK_SIZE; i++) {
            auto cand = Pair{ val[i], idx[i] };
            if (better(cand, best)) {
                best = cand;
            }
        }

        idx[0] = best.idx;
        *next_token_id = best.idx;
    }

    __syncthreads();

    const __nv_bfloat16* embedding = embedding_table + idx[0] * HIDDEN_DIM;
    #pragma unroll
    for (int i = tid; i < HIDDEN_DIM; i += BLOCK_SIZE) {
        next_token_embedding[i] = embedding[i];
    }
}

}

void launch_single_batch_greedy_decode_kernel(
    const __nv_bfloat16* logits,
    const __nv_bfloat16* embedding_table,
    __nv_bfloat16* next_token_embedding,
    int32_t* next_token_id,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    single_batch_greedy_decode_kernel<BLOCK_SIZE>
        <<<1, BLOCK_SIZE, 0, stream>>>(logits, embedding_table, next_token_embedding, next_token_id);
}
