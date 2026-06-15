#include "token_embedding.h"

#include "../llama.h"


namespace {

template <int BLOCK_SIZE> __global__
void token_embedding_kernel(
    const std::int32_t* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embedding_table,
    __nv_bfloat16* __restrict__ hidden_state
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const std::int32_t token_id = token_ids[bid];

    #pragma unroll
    for (int i = tid; i < HIDDEN_DIM; i += BLOCK_SIZE) {
        const int src_idx = token_id * HIDDEN_DIM + i;
        const int dst_idx = bid * HIDDEN_DIM + i;
        hidden_state[dst_idx] = embedding_table[src_idx];
    }
}

} // namespace

void launch_token_embedding_kernel(
    std::size_t token_count,
    const std::int32_t* token_ids,
    const __nv_bfloat16* embedding_table,
    __nv_bfloat16* hidden_state,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;

    token_embedding_kernel<BLOCK_SIZE>
        <<<token_count, BLOCK_SIZE, 0, stream>>>(
        token_ids,
        embedding_table,
        hidden_state
    );
}
