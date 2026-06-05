# Tiny vLLM v1 实现边界
1. 支持 Llama-3.2-1B-Instruct 特定架构
2. 最大 Token 长度 512
3. hidden dim 固定为 2048
4. head dim 固定为 64
5. batch_size = 1

# Note
## cublasGemmEx
`cuBLAS` 假设矩阵是列主元的。`cublasGemmEx` 的计算形式是：$C_\text{col} = \alpha * \text{op}(A) * \text{op}(B) + \beta * C_\text{col}^\text{pre}$

`cublasGemmEx` 入参包含 `transa` 与 `transb`，接收输入 `CUBLAS_OP_T` / `CUBLAS_OP_N` / `CUBLAS_OP_C`。
`cublasGemmEx` 总是按列主元解释内存，`CUBLAS_OP_T` 表示读入矩阵后转置，`CUBLAS_OP_N` 不转置，`CUBLAS_OP_C` 表示对复数矩阵共轭转置。

首先思考一个操作，一个内存中的列主元矩阵 $C_\text{col}$，其规模为 $[L1, L2]$，我们可以将其的一列读成一行，这样一个矩阵就变成了一个行主元矩阵，且这个矩阵的规模为 $[L2, L1]$，相当于对做了一次矩阵转置。

如果要计算 $C = A \times B^T$，在 `cublasGemmEx` 计算得到的结果应该是 $C_\text{col}^T$，这样我们可以用之前那个技巧得到 $C$。则在 `cublasGemmEx` 中的计算应该是 $C_\text{col}^T = B_\text{col} \times A_\text{col}$。

设 $A: [M\times K], B: [N\times K]$，则 `cublasGemmEx` 的输入应该是：
```
cublasGemmEx(
    handle,
    CUBLAS_OP_T,   // B_col^T
    CUBLAS_OP_N,   // A_col
    N,             // m: rows of C_col
    M,             // n: cols of C_col
    K,             // k
    &alpha,
    B, Btype, K,   // lda = K, because B_col is K x N
    A, Atype, K,   // ldb = K, because A_col is K x M
    &beta,
    C, Ctype, N,   // ldc = N, because C_col is N x M
    computeType,
    algo
);
```