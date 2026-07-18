#include <iostream>
#include <cuda_runtime.h>

// Simple macro to catch CUDA errors instantly
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void GEMMkernel(long long *a, long long *b, long long *ans) {
    // Correct way to map 2D threads to a 2D matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < 1000 && col < 1000) {
        long long sum = 0; // Use a temporary register for speed
        for (int i = 0; i < 1000; i++) {
            sum += a[row * 1000 + i] * b[i * 1000 + col];
        }
        ans[row * 1000 + col] = sum;
    }
}

int main() {
    // Use dynamic memory allocation or heap to prevent CPU stack overflow on massive arrays
    static long long a[1000][1000], b[1000][1000], ans[1000][1000];
    
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 1000; j++) {
            a[i][j] = i + j;
            b[i][j] = i + (2 * j);
            ans[i][j] = 0;
        }
    }

    long long *a_gpu, *b_gpu, *ans_gpu;
    cudaCheckError(cudaMalloc((void**)&a_gpu, 1000 * 1000 * sizeof(long long)));
    cudaCheckError(cudaMalloc((void**)&b_gpu, 1000 * 1000 * sizeof(long long)));
    cudaCheckError(cudaMalloc((void**)&ans_gpu, 1000 * 1000 * sizeof(long long)));

    cudaCheckError(cudaMemcpy(a_gpu, a, 1000 * 1000 * sizeof(long long), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(b_gpu, b, 1000 * 1000 * sizeof(long long), cudaMemcpyHostToDevice));

    // Block size: 16x16 threads = 256 threads per block (Safe and highly efficient)
    dim3 blockDim(16, 16);
    // Grid size: Calculate how many blocks of 16 are needed to span 1000 elements
    dim3 gridDim((1000 + blockDim.x - 1) / blockDim.x, (1000 + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    GEMMkernel<<<gridDim, blockDim>>>(a_gpu, b_gpu, ans_gpu);
    
    // Check if the kernel launch itself failed (e.g. invalid configurations)
    cudaCheckError(cudaGetLastError());

    // Make CPU wait and check if execution failed mid-run
    cudaCheckError(cudaDeviceSynchronize());

    // Copy back
    cudaCheckError(cudaMemcpy(ans, ans_gpu, 1000 * 1000 * sizeof(long long), cudaMemcpyDeviceToHost));

    // VERIFICATION: Print a few elements to make sure it computed successfully
    std::cout << "Verification Check (Top-left 3x3 matrix result):" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << ans[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(ans_gpu);
    return 0;
}
