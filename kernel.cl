__kernel void vector_add(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         const int n)
{
    int id = get_global_id(0);
    if (id < n) {
        C[id] = A[id] + B[id];
    }
}

__kernel void matmul(__global const float* A,
                     __global const float* B,
                     __global float* C,
                     const int N,
                     const int M,
                     const int K)
{
    // C размером N x K
    // A размером N x M
    // B размером M x K

    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < N && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}
