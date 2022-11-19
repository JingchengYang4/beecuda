#include <iostream>

#define N 10000000

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    cudaDeviceSynchronize();

    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_out, sizeof(float) * N);
    cudaMemcpy(d_out, out, sizeof(float) * N, cudaMemcpyHostToDevice);


    // Main function
    vector_add<<<1,1>>>(d_out, d_a, d_b, N);

    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    std::cout << "I am editing from visual studio code!!!" << std::endl;
    std::cout << out[0] << std::endl;
}