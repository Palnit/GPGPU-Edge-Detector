//
// Created by Balint on 2023. 11. 12..
//

#ifndef GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_
#define GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_

#include <cuda_runtime.h>
#include <cstdint>

__global__ void convertToGreyScale(uint8_t*);
void test(dim3, dim3, uint8_t*, int);
#endif //GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_
