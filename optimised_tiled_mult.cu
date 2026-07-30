#include <iostream>
#include <chrono>
#include <cuda_profiler_api.h>

using namespace std;

__global__ void tiledGEMMKernel(float *a, float *b, float *ans, int matSize, int tileSize){

    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    extern __shared__ float sharedMemory[];
    float *a_0 = sharedMemory;
    float *b_0 = sharedMemory + tileSize*tileSize;
    float *a_1 = sharedMemory + 2*tileSize*tileSize;
    float *load=a_1, *compute=a_0;

    if(row>=matSize){
        a_0[threadIdx.y*tileSize + threadIdx.x] =0;
    }
    else{
        a_0[threadIdx.y*tileSize + threadIdx.x] = a[row*matSize + threadIdx.x];
    }
    if(col>=matSize){
        b_0[threadIdx.y*tileSize + threadIdx.x] = 0;
    }
    else{
        b_0[threadIdx.y*tileSize + threadIdx.x] = b[threadIdx.y*matSize + col];
    }
    float curr =0;
    for(int i=1;i<= (int)(ceil(matSize/(tileSize*1.0)));i++){
        if(row>=matSize || i*tileSize +threadIdx.x>=matSize ){
            load[threadIdx.y*tileSize + threadIdx.x] = 0;
        }
        else{
            load[threadIdx.y*tileSize + threadIdx.x] =a[row*matSize+tileSize*i+threadIdx.x]; 
        }
        if(col>=matSize || (i*tileSize + threadIdx.y)>=matSize){
            load[tileSize*tileSize + threadIdx.y*tileSize + threadIdx.x] = 0;
        }
        else{
            load[tileSize*tileSize + threadIdx.y*tileSize + threadIdx.x] = b[(i*tileSize+threadIdx.y)*matSize + col];
        }
        __syncthreads();

        for(int j=0;j<tileSize;j++){
            curr+= compute[threadIdx.y*tileSize+j]*compute[tileSize*tileSize + j*tileSize + threadIdx.x];
        }
        compute = load;
        if(load==a_0){
            load=a_1;
        }
        else{
            load = a_0;
        }
        __syncthreads();
    }
    if(row<matSize && col<matSize){
        ans[row*matSize + col] = curr;
    }

}

void benchmark(int matDim,bool output){

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if(output){
        cout<< "Device name: "<<prop.name<<endl;
        cout<<"Max threads per block: "<<prop.maxThreadsPerBlock<<endl;
        cout<<"Shared memory per block: "<<prop.sharedMemPerBlock /1024.0 <<" KB"<<endl;
    }

    float *a = new float [matDim*matDim];
    float *ans = new float [matDim*matDim];
    float *b = new float [matDim*matDim];

    for(int i=0;i<matDim; i++){ //initialization
        for(int j=0;j<matDim;j++){
            a[i*matDim + j] = 1.0;
            b[i*matDim + j] = 1.1;
        }
    }

    float *a_gpu,*b_gpu, *ans_gpu;
    int sizeAllocated = matDim*matDim*sizeof(float);

    auto startTime = chrono::high_resolution_clock::now();
    cudaMalloc((void**)&a_gpu,sizeAllocated);
    cudaMalloc((void**)&b_gpu,sizeAllocated);
    cudaMalloc((void**)&ans_gpu,sizeAllocated);

    cudaMemcpy(a_gpu,a,sizeAllocated,cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu,b,sizeAllocated,cudaMemcpyHostToDevice);

    unsigned int dimBlock = 16;
    unsigned int dimGrid = ceil(matDim/(dimBlock*1.0));
    long long int tileSize = dimBlock;

    dim3 gridDim = {dimGrid,dimGrid};
    dim3 blockDim = {dimBlock,dimBlock};

    size_t sharedMemBytes = 4*tileSize*tileSize*sizeof(float);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tiledGEMMKernel <<<gridDim,blockDim,sharedMemBytes>>> (a_gpu,b_gpu,ans_gpu,matDim,tileSize); 
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    cudaMemcpy(ans, ans_gpu, sizeAllocated, cudaMemcpyDeviceToHost);

    float gpuComputeTime = 0;
    cudaEventElapsedTime(&gpuComputeTime, start, stop);
    auto stopTime = chrono::high_resolution_clock::now();
    chrono::duration<float,milli> duration = stopTime - startTime;
    if(output){
        cout<<"GPU Computation time: "<< gpuComputeTime<<" ms"<<endl;
        cout<<"Total Elapsed Time: "<<duration.count()<<" ms"<<endl;

        long double noOps = 2.0*matDim*matDim*matDim;
        cout<<"GPU Performance: "<<((noOps/gpuComputeTime)*1000.0)/1e9<<" GFLOPS"<<endl;
    }

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(ans_gpu);

    //VERIFICATION STEPS

    // bool flg = true;
    // startTime = chrono::high_resolution_clock::now();
    // for(int i=0;i<matDim;i++){
    //     for(int j=0;j<matDim;j++){
    //         float curr =0;
    //         for(int k=0;k<matDim;k++){
    //             curr+= a[i*matDim +k]*b[k*matDim+j];
    //         }
    //         if(curr!=ans[i*matDim+j]){
    //             flg=false;
    //         }
    //     }
    // }
    // stopTime = chrono::high_resolution_clock::now();
    // duration  =stopTime-startTime;
    // if(output){
    //     cout<<"CPU Computation Time: "<<duration.count()<<" ms"<<endl;
    //     cout<<"CPU Time to GPU Time: "<<duration.count()/gpuComputeTime<<" times!"<<endl;
    //     if(flg){
    //         cout<<"Verified. Computation correct"<<endl;
    //     }
    //     else{
    //         cout<<"Incorrect. Try again"<<endl;
    //     }
    // }
    free(a);
    free(b);
    free(ans);


}

int main(){
    int matDim;
    cout<<"Enter matrix dimension: "<<endl;
    cin>>matDim;
    // for(int i=0;i<18;i++){
    //     benchmark(matDim, false);
    // }
    cudaProfilerStart();
    benchmark(matDim,true);
    cudaProfilerStop();
}