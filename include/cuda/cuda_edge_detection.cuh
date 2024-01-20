//
// Created by Palnit on 2023. 11. 12.
//

#ifndef GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_
#define GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_

#include <cstdint>
#include <cuda_runtime.h>

__global__ void convertToGreyScale(uint8_t*, float*, int, int);
__global__ void GetGaussian(float*, int, float);
__global__ void GaussianFilter(float*, float*, float*, int, int, int);
__global__ void CopyBack(uint8_t*, float*, int, int);
__global__ void DetectionOperator(float*, float*, float*, int, int);
__global__ void NonMaximumSuppression(float*, float*, float*, int, int);
__global__ void DoubleThreshold(float*, float*, int, int, float, float);
__global__ void Hysteresis(float*, float*, int, int);

class CudaDetector {
public:
    CudaDetector(uint8_t* src,
                 int w,
                 int h,
                 int gaussKernelSize,
                 float standardDeviation,
                 float high,
                 float low)
        : m_src(src),
          m_w(w),
          m_h(h),
          m_gaussKernelSize(gaussKernelSize),
          m_standardDeviation(standardDeviation),
          m_high(high),
          m_low(low) {
        CannyEdgeDetection();
    }
private:
    uint8_t* m_src;
    int m_w;
    int m_h;
    int m_gaussKernelSize;
    float m_standardDeviation;
    float m_high;
    float m_low;
    void CannyEdgeDetection();
};
#endif //GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_
