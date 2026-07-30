#include <iostream>
#include <chrono>

using namespace std;



__global__ void threadCoarseningKernel(float *a,float *b, float *ans,int tileSize, int matSize){
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    extern __shared__ float sharedMemory[];
}

int main(){


}