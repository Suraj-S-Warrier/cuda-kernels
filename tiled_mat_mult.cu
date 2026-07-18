#include <iostream>
#include <vector> 
#include <chrono>
#include <cuda_runtime.h>

using namespace std;


__global__ void tiledMatMultKernel(long long *a,long long *b,long long *ans, long long matSize, long long tileSize){
    extern __shared__ long long sharedMem []; //only one shared memory per kernel apparently
    long long *a_shared = sharedMem;
    long long *b_shared = sharedMem + (tileSize*tileSize);

    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    int phases = ceil(matSize/(tileSize*1.0));

    long long curr =0;
    for(int i=0;i<phases;i++){ // iterating through our phases
        if(i*tileSize + threadIdx.y>=matSize || row>=matSize){
            a_shared[threadIdx.x*tileSize + threadIdx.y]=0;
        }
        else{
            a_shared[threadIdx.x*tileSize + threadIdx.y] = a[row*matSize + i*tileSize+ threadIdx.y];
        }
        
        if(i*tileSize + threadIdx.x >= matSize || col>=matSize){
            b_shared[threadIdx.x*tileSize + threadIdx.y]=0;
        }
        else{
            b_shared[threadIdx.x*tileSize + threadIdx.y] = b[(i*tileSize + threadIdx.x)*matSize + col];
        }
        __syncthreads(); //to make sure all threads together has loaded the entire tile
        for(int j=0;j<tileSize;j++){
            curr += a_shared[threadIdx.x*tileSize + j]*b_shared[j*tileSize+threadIdx.y];
        }
        __syncthreads(); // to make sure each thread computed the curr value before loading the next tile
    }

    if(row<matSize && col<matSize){
        ans[row*matSize + col] = curr;
    }

}

int main(){

    int matDim;
    cout<<"Enter matrix dimension: "<<endl;
    cin>>matDim;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout<< "Device name: "<<prop.name<<endl;
    cout<<"Max threads per block: "<<prop.maxThreadsPerBlock<<endl;
    cout<<"Shared memory per block: "<<prop.sharedMemPerBlock /1024.0 <<" KB"<<endl;

    long long *a = new long long [matDim*matDim];
    long long *b = new long long [matDim* matDim];
    long long *ans = new long long [matDim*matDim];

    for(int i=0;i<matDim; i++){ //initialization
        for(int j=0;j<matDim;j++){
            a[i*matDim + j] = i+j;
            b[i*matDim + j] = i + 2*j;
        }
    }

    long long *a_gpu,*b_gpu, *ans_gpu;
    long long sizeAllocated = matDim*matDim*sizeof(long long);

    auto startTime = chrono::high_resolution_clock::now();
    cudaMalloc((void**)&a_gpu,sizeAllocated);
    cudaMalloc((void**)&b_gpu,sizeAllocated);
    cudaMalloc((void**)&ans_gpu,sizeAllocated);

    cudaMemcpy(a_gpu,a,sizeAllocated,cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu,b,sizeAllocated,cudaMemcpyHostToDevice);

    unsigned int dimBlock = 32;
    unsigned int dimGrid = ceil(matDim/(dimBlock*1.0));
    long long int tileSize = dimBlock;

    dim3 gridDim = {dimGrid,dimGrid};
    dim3 blockDim = {dimBlock,dimBlock};

    size_t sharedMemBytes = 2*tileSize*tileSize*sizeof(long long);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tiledMatMultKernel <<<gridDim,blockDim,sharedMemBytes>>> (a_gpu,b_gpu,ans_gpu,matDim,tileSize); 
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    cudaMemcpy(ans, ans_gpu, sizeAllocated, cudaMemcpyDeviceToHost); //copy answer back to CPU mem

    float gpuComputeTime = 0;
    cudaEventElapsedTime(&gpuComputeTime, start, stop);
    auto stopTime = chrono::high_resolution_clock::now();
    chrono::duration<float,milli> duration = stopTime - startTime;
    cout<<"GPU Computation time: "<< gpuComputeTime<<" ms"<<endl;
    cout<<"Total Elapsed Time: "<<duration.count()<<" ms"<<endl;

    long double noOps = 2.0*matDim*matDim*matDim;
    cout<<"GPU Performance: "<<((noOps/gpuComputeTime)*1000.0)/1e9<<" GFLOPS"<<endl;

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(ans_gpu);

    //VERIFICATION STEPS

    bool flg = true;
    startTime = chrono::high_resolution_clock::now();
    for(int i=0;i<matDim;i++){
        for(int j=0;j<matDim;j++){
            long long curr =0;
            for(int k=0;k<matDim;k++){
                curr+= a[i*matDim +k]*b[k*matDim+j];
            }
            if(curr!=ans[i*matDim+j]){
                flg=false;
            }
        }
    }
    stopTime = chrono::high_resolution_clock::now();
    duration  =stopTime-startTime;
    cout<<"CPU Computation Time: "<<duration.count()<<" ms"<<endl;
    cout<<"CPU Time to GPU Time: "<<duration.count()/gpuComputeTime<<" times!"<<endl;
    if(flg){
        cout<<"Verified. Computation correct"<<endl;
    }
    else{
        cout<<"Incorrect. Try again"<<endl;
    }
    free(a);
    free(b);
    free(ans);





    return 0;
}