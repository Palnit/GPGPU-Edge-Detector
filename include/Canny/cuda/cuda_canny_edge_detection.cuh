//
// Created by Palnit on 2023. 11. 12.
//

#ifndef GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_
#define GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_

#include <cstdint>
#include <cuda_runtime.h>
#include "include/Canny/canny_timings.h"

__global__ void DetectionOperator(float*, float*, float*, int, int);
__global__ void NonMaximumSuppression(float*, float*, float*, int, int);
__global__ void DoubleThreshold(float*, float*, int, int, float, float);
__global__ void Hysteresis(float*, float*, int, int);

class CudaCannyDetector {
public:
    CudaCannyDetector(uint8_t* src,
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
    CannyTimings GetTimings() {
        return m_timings;
    }

private:
    struct CudaTimers {
        CudaTimers() {
            cudaEventCreate(&GrayScale_start);
            cudaEventCreate(&GrayScale_stop);
            cudaEventCreate(&GaussCreation_start);
            cudaEventCreate(&GaussCreation_stop);
            cudaEventCreate(&Blur_start);
            cudaEventCreate(&Blur_stop);
            cudaEventCreate(&SobelOperator_start);
            cudaEventCreate(&SobelOperator_stop);
            cudaEventCreate(&NonMaximumSuppression_start);
            cudaEventCreate(&NonMaximumSuppression_stop);
            cudaEventCreate(&DoubleThreshold_start);
            cudaEventCreate(&DoubleThreshold_stop);
            cudaEventCreate(&Hysteresis_start);
            cudaEventCreate(&Hysteresis_stop);
            cudaEventCreate(&All_start);
            cudaEventCreate(&All_stop);
        }
        ~CudaTimers() {
            cudaEventDestroy(GrayScale_start);
            cudaEventDestroy(GrayScale_stop);
            cudaEventDestroy(GaussCreation_start);
            cudaEventDestroy(GaussCreation_stop);
            cudaEventDestroy(Blur_start);
            cudaEventDestroy(Blur_stop);
            cudaEventDestroy(SobelOperator_start);
            cudaEventDestroy(SobelOperator_stop);
            cudaEventDestroy(NonMaximumSuppression_start);
            cudaEventDestroy(NonMaximumSuppression_stop);
            cudaEventDestroy(DoubleThreshold_start);
            cudaEventDestroy(DoubleThreshold_stop);
            cudaEventDestroy(Hysteresis_start);
            cudaEventDestroy(Hysteresis_stop);
            cudaEventDestroy(All_start);
            cudaEventDestroy(All_stop);
        }
        cudaEvent_t GrayScale_start;
        cudaEvent_t GrayScale_stop;
        cudaEvent_t GaussCreation_start;
        cudaEvent_t GaussCreation_stop;
        cudaEvent_t Blur_start;
        cudaEvent_t Blur_stop;
        cudaEvent_t SobelOperator_start;
        cudaEvent_t SobelOperator_stop;
        cudaEvent_t NonMaximumSuppression_start;
        cudaEvent_t NonMaximumSuppression_stop;
        cudaEvent_t DoubleThreshold_start;
        cudaEvent_t DoubleThreshold_stop;
        cudaEvent_t Hysteresis_start;
        cudaEvent_t Hysteresis_stop;
        cudaEvent_t All_start;
        cudaEvent_t All_stop;
    };

    uint8_t* m_src;
    int m_w;
    int m_h;
    int m_gaussKernelSize;
    float m_standardDeviation;
    float m_high;
    float m_low;
    CudaTimers m_timers;
    CannyTimings m_timings;
    void CannyEdgeDetection();
};
#endif //GPGPU_EDGE_DETECTOR_CUDA_INCLUDE_CUDA_EDGE_DETECTION_CUH_
