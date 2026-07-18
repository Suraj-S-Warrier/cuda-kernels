//Press Ctrl+Shift+B to compile in the project...this is coz tasks.json has been added to .vscode folder
//since nvidia doesnt automatically recognise the cl.exe file which we have, which is of 2026 VS...it needs 2022 VS

//add events to track accurate time 
// add profiling tools (gives memory throughput, occupancy)
// separate kernel execution time from total end-to-end time
//calculate GFLOPS
//for different sizes too( make sizes depend on variables rather than hardcoded nos)
//print hardware metadata

#include<iostream>
#include<chrono>
#include<cuda_runtime.h>

#define matSize 2048
using namespace std;

__global__ void GEMMkernel(long long *a,long long *b, long long *ans){
    long long row = blockIdx.x*blockDim.x+threadIdx.x,col = blockIdx.y*blockDim.y+threadIdx.y;
    if(row>=matSize || col>=matSize){
        return;
    }

    for(int i=0;i<matSize;i++){
        ans[row*matSize+col] += a[row*matSize+i]*b[i*matSize+col]; //gpu must always treat the arrays as flat arrays
    }
}
int main(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout<< "Device name: "<<prop.name<<endl;
    cout<<"Max threads per block: "<<prop.maxThreadsPerBlock<<endl;
    cout<<"Shared memory per block: "<<prop.sharedMemPerBlock /1024.0 <<" KB"<<endl;

    static long long a[matSize][matSize],b[matSize][matSize], ans[matSize][matSize], verify[matSize][matSize];
    for(int i=0;i<matSize;i++){
        for(int j=0;j<matSize;j++){
            a[i][j] = i+j;
            b[i][j] = i+(2*j);
            ans[i][j]=0;
            verify[i][j]=0;
        }
    }

    //allocate space on gpu
    long long *a_gpu,*b_gpu,*ans_gpu;
    auto start_time = chrono::high_resolution_clock::now(); // to track the total time
    cudaMalloc((void**)&a_gpu,matSize*matSize*sizeof(long long));
    cudaMalloc((void**)&ans_gpu,matSize*matSize*sizeof(long long));
    cudaMalloc((void**)&b_gpu,matSize*matSize*sizeof(long long));

    //transfer to gpu memory
    cudaMemcpy(a_gpu,a,matSize*matSize*sizeof(long long),cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu,b,matSize*matSize*sizeof(long long),cudaMemcpyHostToDevice);
    cudaMemcpy(ans_gpu,ans,matSize*matSize*sizeof(long long),cudaMemcpyHostToDevice);

    //launch kernel
    unsigned int block_dim = 32;
    unsigned int grid_dim = ceil(matSize/(block_dim*1.0));
    dim3 gridDim = {grid_dim, grid_dim};
    dim3 blockDim = {block_dim,block_dim};

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    
    cudaEventRecord(start);
    GEMMkernel<<<gridDim,blockDim>>>(a_gpu,b_gpu,ans_gpu);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    //make cpu wait for ans
    // cudaDeviceSynchronize();
    auto stop_time = chrono::high_resolution_clock::now();
    chrono::duration<float,milli> duration = stop_time-start_time;
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start, stop);
    
    cout<<"GPU Time: "<<milliseconds<<" ms"<<endl;
    cout<<"Total End-to-End Time: "<<duration.count()<<" ms"<<endl;
    double noOps = 2.0*matSize*matSize*matSize; // for mat mult, it is 2N^3
    cout<<"GPU performance: "<< (noOps/(milliseconds/1000.0))/1e9<<" GFLOPS"<<endl;

    //get answer to cpu memory
    cudaMemcpy(ans,ans_gpu, matSize*matSize*(sizeof(long long)),cudaMemcpyDeviceToHost);
    //free the space
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(ans_gpu);

    //doing sequential
    start_time = chrono::high_resolution_clock::now();
    for(int i=0;i<matSize;i++){
        for(int j=0;j<matSize;j++){
            for(int k=0;k<matSize;k++){
                verify[i][j]+=(a[i][k]*b[k][j]);
            }
        }
    }
    stop_time = chrono::high_resolution_clock::now();
    duration = stop_time-start_time;
    cout<<"CPU Time: "<<duration.count()<<" ms"<<endl;
    cout<<"CPU Time to GPU time: "<< duration.count()/milliseconds<<" times!"<<endl;

    //verification
    bool correct = true;
    for(int i=0;i<matSize;i++){
        for(int j=0;j<matSize;j++){
            if(ans[i][j] != verify[i][j]){
                correct = false;
            }
        }
    }
    if(correct){
        cout<<"Verified. Computation correct"<<endl;
    }
    else{
        cout<<"Verified. Computation incorrect. Retry"<<endl;
    }


    
}