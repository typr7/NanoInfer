#include "rope.h"
#include "llama.h"


__global__
void RoPEKernel(
    int ldq,
    int ldk,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k
) {
    const int qidx = blockIdx.x * ldq + 2 * threadIdx.x;
    const int double_i = 2 * (threadIdx.x % (HEAD_DIM / 2));
    const float theta = blockIdx.x / powf(500000.f, static_cast<float>(double_i) / HEAD_DIM);
    const float cos_theta = cosf(theta);
    const float sin_theta = sinf(theta);
    if (2 * threadIdx.x < Q_PROJ_DIM) {
        const float val1 = q[qidx];
        const float val2 = q[qidx + 1];
        q[qidx] = val1 * cos_theta - val2 * sin_theta;
        q[qidx + 1] = val1 * sin_theta + val2 * cos_theta;
    }
    const int kidx = blockIdx.x * ldk + 2 * threadIdx.x;
    if (2 * threadIdx.x < K_PROJ_DIM) {
        const float val1 = k[kidx];
        const float val2 = k[kidx + 1];
        k[kidx] = val1 * cos_theta - val2 * sin_theta;
        k[kidx + 1] = val1 * sin_theta + val2 * cos_theta;
    }
}

void launchRoPEKernel(
    std::size_t token_num,
    int ldq,
    int ldk,
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    cudaStream_t stream
) {
    RoPEKernel<<<token_num, HIDDEN_DIM / 2, 0, stream>>>(ldq, ldk, q, k);
}
