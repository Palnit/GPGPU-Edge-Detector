//
// Created by Balint on 2023. 11. 12..
//

#include "cuda_include/cuda_edge_detection.cuh"
#include <cstdio>
#include <cstdint>

typedef struct RGBA {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
} RGBA;

__global__ void convertToGreyScale(uint8_t* asd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    RGBA* asd2 = (RGBA*) (int32_t*) ((uint8_t*) asd);
    std::printf("%d,%d,%d,%d,%d\n", idx, asd2->r, asd2->g, asd2->b, asd2->a);
}

void test(int a, int b, uint8_t* asd) {
    convertToGreyScale<<<a, b>>>(asd);
    cudaDeviceSynchronize();
}