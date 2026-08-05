#include <iostream>
#include <chrono>
#include <cuda_profiler_api.h>

using namespace std;


__global__ void convolution(float *a, int width, int height,float *filter, int radius, float *ans){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    float val=0.0f;
    for(int i=row-radius;i<=row+radius;i++){
        for(int j=col-radius;j<=col+radius;j++){
            if(!(row<0 || row>=height || i<0 || i>=height || col<0 || col>=width || j<0 || j>=width) ){
                val += filter[(i-(row-radius))*(2*radius +1) + (j-(col-radius))]*a[i*width + j];
            }
        }
    }
    if(row<height && col<width){
        ans[row*width+col] = val;
    }


}

void benchmark(int height, int width,int radius, bool output){

    float *a = new float [height*width];
    float *filter = new float [(2*radius +1)*(2*radius+1)];
    float *ans = new float [height*width];
    float* verify = new float [height*width];

    for(int i=0;i<height; i++){ //initialization
        for(int j=0;j<width;j++){
            a[i*width + j] = 1.1 + (0.01*j);
        }
    }

    for(int i=0;i<2*radius+1;i++){
        for(int j=0;j<2*radius+1;j++){
            filter[i*(2*radius+1) + j]=1.1;
        }
    }

    float *a_gpu, *filter_gpu, *ans_gpu;
    auto startTime = chrono::high_resolution_clock::now();
    cudaMalloc((void**)&a_gpu, sizeof(float)*height*width);
    cudaMalloc((void**)&filter_gpu, sizeof(float)*(2*radius +1)*(2*radius+1));
    cudaMalloc((void**)&ans_gpu, sizeof(float)*height*width);

    cudaMemcpy(a_gpu, a, sizeof(float)*height*width, cudaMemcpyHostToDevice);
    cudaMemcpy(filter_gpu, filter, sizeof(float)*(2*radius +1)*(2*radius+1), cudaMemcpyHostToDevice);
    // cudaMemcpy(ans_gpu, ans, sizeof(float)*height*width, cudaMemcpyHostToDevice);

    unsigned int blockSize = 16;
    unsigned int gridWidth = ceil(width/(blockSize*1.0));
    unsigned int gridHeight = ceil(height/(blockSize*1.0));
    dim3 blockDim = {blockSize,blockSize};
    dim3 gridDim = {gridWidth,gridHeight};

    //size_t sharedMemorySize = blockSize*blockSize*sizeof(float);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    convolution<<<gridDim,blockDim>>> (a_gpu,width,height,filter_gpu,radius,ans_gpu);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    cudaMemcpy(ans,ans_gpu,sizeof(float)*height*width,cudaMemcpyDeviceToHost);

    float gpuComputeTime = 0;
    cudaEventElapsedTime(&gpuComputeTime, start, stop);
    auto stopTime = chrono::high_resolution_clock::now();
    chrono::duration<float,milli> duration = stopTime - startTime;
    if(output){
        cout<<"GPU Computation time: "<< gpuComputeTime<<" ms"<<endl;
        cout<<"Total Elapsed Time: "<<duration.count()<<" ms"<<endl;
    }

    cudaFree(a_gpu);
    cudaFree(filter_gpu);
    cudaFree(ans_gpu);

    if(output){
        bool correct = true;
        startTime = chrono::high_resolution_clock::now();
        for(int i=0;i<height;i++){
            for(int j=0;j<width;j++){
                float temp=0.0f;
                for(int k=i-radius;k<=i+radius;k++){
                    for(int l=j-radius;l<=j+radius;l++){
                        if(k>=0 && k<height && l>=0 && l<width){
                            temp += filter[(k-(i-radius))*(2*radius+1) + l-(j-radius)]*a[k*width+l];
                        }
                    }
                }
                if(abs(ans[i*(width)+j]-temp) >0.01){
                    correct = false;
                    cout<<"Incorrect result. Check again."<<endl;
                    return;

                }
            }
        }
        stopTime = chrono::high_resolution_clock::now();
        duration  =stopTime-startTime;
        if(output){
            cout<<"CPU Computation Time: "<<duration.count()<<" ms"<<endl;
            cout<<"CPU Time to GPU Time: "<<duration.count()/gpuComputeTime<<" times!"<<endl;
        }

    }




}

int main(){
    int n,m,r;
    cout<<"Enter height and width"<<endl;
    cin>>n>>m;
    cout<<"Enter filter radius"<<endl;
    cin>>r;
    for(int i=0;i<9;i++){
        benchmark(n,m,r,false);
    }
    cudaProfilerStart();
    benchmark(n,m,r,true);
    cudaProfilerStop();
    return 0;
}