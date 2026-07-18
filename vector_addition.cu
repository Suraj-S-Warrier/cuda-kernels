#include<cuda_runtime.h>
#include<iostream>
#include<chrono>
#include<vector>

using namespace std;

__global__ void vectorMultKernel(long long a[],long long b[],long long ans[]){
    int i = blockIdx.x * 512 + threadIdx.x;
    if(i>=1000000){
        return;
    }
    ans[i] = a[i]+b[i];

}

__host__ int main(){
    //vector addition

    int N = 1000000;
    vector<long long> a(N), b(N), ans(N,0), verify(N,0);
    for(int i=0;i<1000000;i++){
        a[i]=i;
        b[i] = 1000000-i;
        ans[i]=0;
    }

    long long *a_gpu, *b_gpu, *ans_gpu;
    cudaMalloc((void **)&a_gpu, 1000000*sizeof(long long));
    cudaMalloc((void **)&b_gpu, 1000000*sizeof(long long));
    cudaMalloc((void **)&ans_gpu, 1000000*sizeof(long long));

    cudaMemcpy(a_gpu, a.data(),1000000*sizeof(long long),cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b.data(),1000000*sizeof(long long),cudaMemcpyHostToDevice);
    
    dim3 gridDim = {1955};
    dim3 blockDim = {512};

    auto start = chrono::high_resolution_clock::now();
    vectorMultKernel<<<gridDim,blockDim>>> (a_gpu,b_gpu,ans_gpu);

    cudaDeviceSynchronize();
    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<float,milli> duration = stop-start;
    cout<<"GPU Time: "<<duration.count()<<" ms"<<endl;

    cudaMemcpy(ans.data(), ans_gpu,1000000*sizeof(long long),cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(ans_gpu);
    start = chrono::high_resolution_clock::now();
    for(int i=0;i<1000000;i++){
        verify[i] = a[i]+b[i];
    }
    stop = chrono::high_resolution_clock::now();
    duration = stop-start;
    cout<<"CPU Time: "<<duration.count()<<" ms"<<endl;

    for(int i=0;i<1000000;i++){
        if(ans[i]!=verify[i]){
            cout<<"Verification complete. Incorrect. Retry."<<endl;
            return 0;
        }
    }
    cout<<"Verification complete. Correct"<<endl;



    return 0;
}