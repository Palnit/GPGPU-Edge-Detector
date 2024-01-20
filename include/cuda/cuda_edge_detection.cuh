//
// Created by Palnit on 2023. 11. 12.
//

#ifndef GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_
#define GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_

#include "../../../../../../../../Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include/cuda_runtime.h"
#include "../../../../../../../../Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.33.31629/include/cstdint"

__global__ void convertToGreyScale(uint8_t*, float*, int, int);
__global__ void GetGaussian(float*);
__global__ void GaussianFilter(float*, float*, float*, int, int);
__global__ void CopyBack(uint8_t*, float*, int, int);
__global__ void DetectionOperator(float*, float*, float*, int, int);
__global__ void NonMaximumSuppression(float*, float*, float*, int, int);
__global__ void DoubleThreshold(float*, float*, int, int, float, float);
__global__ void Hysteresis(float*, float*, int, int);
void CannyEdgeDetection(uint8_t*, int, int);
#endif //GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_
