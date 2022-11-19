#include<iostream>

using namespace std;

#define maxBee 1000000

struct bee
{
    int age;
};

struct brood
{
    
};

__global__ void beeFill(bee *b, bee f)
{
    b[blockIdx.x] = f;
};

__global__ void beeGlobalUpdate(bee *b)
{
    b[blockIdx.x].age += 1;
};

int main(void) 
{
    bee *h_bee, *d_bee;

    h_bee = (bee*)malloc(sizeof(bee) * maxBee);
    cudaMalloc((void**) &d_bee, sizeof(bee) * maxBee);
    bee newBee;
    newBee.age = 10;
    beeFill<<<maxBee, 1>>>(d_bee, newBee);
    //cudaMemcpy(d_bee, h_bee, sizeof(bee) * maxBee, cudaMemcpyHostToDevice);

    beeGlobalUpdate<<<maxBee, 1>>>(d_bee);

    cudaMemcpy(h_bee, d_bee, sizeof(bee) * maxBee, cudaMemcpyDeviceToHost);

    cout << (h_bee[0]).age << endl;
    cout << (h_bee[10]).age << endl;
    return 0;
}