//
// Created by Palnit on 2023. 11. 12.
//

#include "include/cuda/cuda_edge_detection.cuh"
#include <cstdio>
#include <cstdint>

typedef struct RGBA {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
} RGBA;

__global__ void convertToGreyScale(uint8_t* asd, int w, int h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    RGBA* color = (RGBA*) (asd + (x * 4) + (y * w * 4));
    color->r = color->g = color->b =
        0.299 * color->r
            + 0.587 * color->g
            + 0.114 * color->b;

}

void test(dim3 a, dim3 b, uint8_t* asd, int w, int h) {
    convertToGreyScale<<<a, b>>>(asd, w, h);
    cudaDeviceSynchronize();
}