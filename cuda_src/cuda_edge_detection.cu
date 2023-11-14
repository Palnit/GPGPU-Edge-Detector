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

__global__ void convertToGreyScale(uint8_t* asd, int n) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idx = y * blockDim.x * gridDim.x + x;
    if (idx > n) {
        return;
    }
    RGBA* color = (RGBA*) (int32_t*) ((uint8_t*) asd + (idx * 4));
    color->r = color->g = color->b =
        0.299 * color->r
            + 0.587 * color->g
            + 0.114 * color->b;
}

void test(dim3 a, dim3 b, uint8_t* asd, int n) {
    convertToGreyScale<<<a, b>>>(asd, n);
    cudaDeviceSynchronize();
}