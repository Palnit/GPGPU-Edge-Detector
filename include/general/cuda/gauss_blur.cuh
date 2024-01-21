//
// Created by Palnit on 2024. 01. 21.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_GAUSS_BLUR_CUH_
#define GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_GAUSS_BLUR_CUH_

#include <cuda_runtime.h>
#include <cstdint>

typedef struct RGBA {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
} RGBA;

__global__ void convertToGreyScale(uint8_t*, float*, int, int);
__global__ void GetGaussian(float*, int, float);
__global__ void GaussianFilter(float*, float*, float*, int, int, int);
__global__ void CopyBack(uint8_t*, float*, int, int);

#endif //GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_GAUSS_BLUR_CUH_
