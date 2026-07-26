#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

//after block level aggregation, we can aggregate across the blocks by either: 
//1) doing an atomicAdd() operation which accesses VRAM memory in atomic fashion
//2) or, save block level sums into another VRAM partial_sums block, and run another kernel to sum that in a similar fashion. But dont make it 
// recursive from kernel...just call kernel in a loop from cpu, rather than calling kernel from kernel since that has high overhead.
//3) or, worst comes to worst, give it to CPU to sum it up. But thats very bad. Coz what do u mean u summed it up in GPU just to give it back to CPU?

__global__ void reductionKernel(double *a,double *ans, int size){

    extern __shared__ double partialAns[];
    int ind = blockIdx.x*blockDim.x + threadIdx.x;
    if(ind <size){
        partialAns[threadIdx.x] = a[blockIdx.x*blockDim.x + threadIdx.x];
    }
    else{
        partialAns[threadIdx.x] = 0;
    }
    
    
    for(int stride=1;stride<blockDim.x;stride*=2){
        __syncthreads();
        if(threadIdx.x%(2*stride)==0 && (threadIdx.x+stride < blockDim.x)){
            partialAns[threadIdx.x]+= partialAns[threadIdx.x+stride];
        }
    }
    if(threadIdx.x==0){
        ans[blockIdx.x] = partialAns[threadIdx.x];
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
        int numberOfBlocks = ceil(n_gpu/(blockLength*1.0));
        unsigned int gridLength = ceil(n_gpu/(blockLength*1.0));
        dim3 gridDim = {gridLength,1};
        reductionKernel<<<gridDim,blockDim,sharedMemorySize>>> (src_buffer,dest_buffer,n_gpu);
        // cudaMemcpy(a_gpu,ans_gpu,numberOfBlocks*sizeof(float),cudaMemcpyDeviceToDevice);
        swap(src_buffer,dest_buffer);
        n_gpu=numberOfBlocks;
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
    startTime = chrono::high_resolution_clock::now();
    double verify=0;
    for(int i=0;i<n;i++){
        verify+=a[i];
    }
    stopTime = chrono::high_resolution_clock::now();
    duration  =stopTime-startTime;
    if(output){
        cout<<"CPU Computation Time: "<<duration.count()<<" ms"<<endl;
        cout<<"CPU Time to GPU Time: "<<duration.count()/gpuComputeTime<<" times!"<<endl;
    }
    if(fabs(verify-finalAns)>(1)){
        if(output){
            cout<<verify<<" and "<<finalAns<<" are not equal."<<endl;
            cout<<"Incorrect. Try again"<<endl;
        }
    }
    else{
        if(output){
            cout<<"Verified. Computation correct"<<endl;
        }
    }

    free(a);
    
}

int main(){

    cout<<"Enter size of array: "<<endl;
    int n;
    cin>>n;
    for(int i=0;i<17;i++){
        benchmark(n,false);   // to warm up the gpu and cpu, and initialise everything
    }
    benchmark(n,true);

    return 0;
}