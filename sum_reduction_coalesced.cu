#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

using namespace std;

__global__ void reductionCoalescedKernel(double *a, double *ans, int size){

    int ind = 2*blockIdx.x*blockDim.x + threadIdx.x;
    extern __shared__ double sharedMemory[];

    if(ind>=size){
        sharedMemory[threadIdx.x]=0;
    }
    else if((2*blockIdx.x+1)*blockDim.x +threadIdx.x <size){
        sharedMemory[threadIdx.x] = a[ind] + a[ind+blockDim.x];
    }
    else{

        sharedMemory[threadIdx.x] = a[ind];
    }

    int dimBlock = blockDim.x;
    while(dimBlock>1){
        __syncthreads();
        if(threadIdx.x<= (dimBlock-1)/2 && threadIdx.x+1+((dimBlock-1)/2)<dimBlock){
            sharedMemory[threadIdx.x] += sharedMemory[threadIdx.x+1+((dimBlock-1)/2)];
        }
        dimBlock = (dimBlock+1)/2;
    }
    if(threadIdx.x==0){
        ans[blockIdx.x] = sharedMemory[threadIdx.x];
    }
    
}

void benchmark(int n,bool output){

    int n_gpu;
    n_gpu=n;
    double *a = new double [n];
    double finalAns =0;
    for(int i=0;i<n;i++){
        a[i] = i*1.1;
    }
    double *a_gpu,*ans_gpu;
    double *src_buffer, *dest_buffer;

    auto startTime = chrono::high_resolution_clock::now();
    cudaMalloc((void**)&a_gpu, n*sizeof(double));
    cudaMalloc((void**)&ans_gpu, n*sizeof(double));
    cudaMemcpy(a_gpu,a,n*sizeof(double),cudaMemcpyHostToDevice);
    src_buffer = a_gpu;
    dest_buffer = ans_gpu;

    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    unsigned int blockLength = 512;
    
    dim3 blockDim = {blockLength,1};
    

    size_t sharedMemorySize = sizeof(double)*blockLength;
    cudaEventRecord(start);
    while(n_gpu>1){
        unsigned int gridLength = ceil(n_gpu/(blockLength*2.0));
        dim3 gridDim = {gridLength,1};
        reductionCoalescedKernel<<<gridDim,blockDim,sharedMemorySize>>> (src_buffer,dest_buffer,n_gpu);
        // cudaMemcpy(a_gpu,ans_gpu,numberOfBlocks*sizeof(float),cudaMemcpyDeviceToDevice);
        swap(src_buffer,dest_buffer);
        n_gpu=gridLength;
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaMemcpy(&finalAns,src_buffer,sizeof(double),cudaMemcpyDeviceToHost);
    float gpuComputeTime = 0;
    cudaEventElapsedTime(&gpuComputeTime, start, end);
    auto stopTime = chrono::high_resolution_clock::now();
    chrono::duration<float,milli> duration = stopTime - startTime;
    if(output){
        cout<<"GPU Computation time: "<< gpuComputeTime<<" ms"<<endl;
        cout<<"Total Elapsed Time: "<<duration.count()<<" ms"<<endl;
    }

    double noOps = n-1;
    if(output){
        cout<<"GPU Performance: "<<((noOps/gpuComputeTime)*1000.0)/1e9<<" GFLOPS"<<endl;
    }
    cudaFree(a_gpu);
    cudaFree(ans_gpu);

    //VERIFICATION STEPS
    // startTime = chrono::high_resolution_clock::now();
    // double verify=0;
    // for(int i=0;i<n;i++){
    //     verify+=a[i];
    // }
    // stopTime = chrono::high_resolution_clock::now();
    // duration  =stopTime-startTime;
    // if(output){
    //     cout<<"CPU Computation Time: "<<duration.count()<<" ms"<<endl;
    //     cout<<"CPU Time to GPU Time: "<<duration.count()/gpuComputeTime<<" times!"<<endl;
    // }
    // if(fabs(verify-finalAns)>(1)){
    //     if(output){
    //         cout<<verify<<" and "<<finalAns<<" are not equal."<<endl;
    //         cout<<"Incorrect. Try again"<<endl;
    //     }
    // }
    // else{
    //     if(output){
    //         cout<<"Verified. Computation correct"<<endl;
    //     }
    // }

    free(a);
    
}

int main(){

    cout<<"Enter size of array: "<<endl;
    int n;
    cin>>n;
    // for(int i=0;i<17;i++){
    //     benchmark(n,false);   // to warm up the gpu and cpu, and initialise everything
    // }
    cudaProfilerStart();
    benchmark(n,true);
    cudaProfilerStop();

    return 0;
}