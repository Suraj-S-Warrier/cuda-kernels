#include <iostream>
#include <chrono>
#include <cuda_profiler_api.h>

using namespace std;


__global__ void convolution(float *a, int width, int height,float *filter, int radius, float *ans){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.y + threadIdx.x;

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

    float *a_gpu, *filter_gpu, *ans_gpu;
    cudaMalloc((void**)&a_gpu, sizeof(float)*height*width);
    cudaMalloc((void**)&filter_gpu, sizeof(float)*(2*radius +1)*(2*radius+1));
    cudaMalloc((void**)&ans_gpu, sizeof(float)*height*width);

    cudaMemcpy(a_gpu, a, sizeof(float)*height*width, cudaMemcpyHostToDevice);
    cudaMemcpy(filter_gpu, filter, sizeof(float)*(2*radius +1)*(2*radius+1), cudaMemcpyHostToDevice);
    cudaMemcpy(ans_gpu, ans, sizeof(float)*height*width, cudaMemcpyHostToDevice);

    dim3 blockDim = {16,16};
    dim3 gridDim = {};



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