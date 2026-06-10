#include "token_embedding.h"

#include "../llama.h"


namespace {

__global__
void token_embedding_kernel(
    const std::int32_t* __restrict__ token_ids,
    const __nv_bfloat16* __restrict__ embedding_table,
    __nv_bfloat16* __restrict__ hidden_state
) {
    const int dst = blockIdx.x * HIDDEN_DIM + threadIdx.x;
    const int src = token_ids[blockIdx.x] * HIDDEN_DIM + threadIdx.x;
    hidden_state[dst] = embedding_table[src];
    hidden_state[dst + HIDDEN_DIM / 2] = embedding_table[src + HIDDEN_DIM / 2];
}

} // namespace

void launch_token_embedding_kernel(
    std::size_t token_count,
    const std::int32_t* token_ids,
    const __nv_bfloat16* embedding_table,
    __nv_bfloat16* hidden_state,
    cudaStream_t stream
) {
    token_embedding_kernel<<<token_count, HIDDEN_DIM / 2, 0, stream>>>(
        token_ids,
        embedding_table,
        hidden_state
    );
}
