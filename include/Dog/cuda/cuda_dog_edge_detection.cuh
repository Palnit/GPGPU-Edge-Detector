//
// Created by Palnit on 2024. 01. 21.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOG_EDGE_DETECTION_CUH_
#define GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOG_EDGE_DETECTION_CUH_

#include <cstdint>
#include <cuda_runtime.h>
#include "include/Dog/dog_timings.h"

__global__ void DifferenceOfGaussian(float*, float*, float*, int);

class CudaDogDetector {
public:
    CudaDogDetector(uint8_t* src,
                    int w,
                    int h,
                    int gaussKernelSize,
                    float standardDeviation1,
                    float standardDeviation2) :
        m_src(src),
        m_w(w),
        m_h(h),
        m_gaussKernelSize(gaussKernelSize),
        m_standardDeviation1(standardDeviation1),
        m_standardDeviation2(standardDeviation2) {
        DogDetect();
    }
    DogTimings GetTimings() {
        return m_timings;
    }
private:
    struct CudaTimers {
        CudaTimers() {
            cudaEventCreate(&All_start);
            cudaEventCreate(&All_stop);
            cudaEventCreate(&GrayScale_start);
            cudaEventCreate(&GrayScale_stop);
            cudaEventCreate(&Gauss1Creation_start);
            cudaEventCreate(&Gauss1Creation_stop);
            cudaEventCreate(&Gauss2Creation_start);
            cudaEventCreate(&Gauss2Creation_stop);
            cudaEventCreate(&DifferenceOfGaussian_start);
            cudaEventCreate(&DifferenceOfGaussian_stop);
            cudaEventCreate(&Convolution_start);
            cudaEventCreate(&Convolution_stop);
        }
        cudaEvent_t All_start;
        cudaEvent_t All_stop;
        cudaEvent_t GrayScale_start;
        cudaEvent_t GrayScale_stop;
        cudaEvent_t Gauss1Creation_start;
        cudaEvent_t Gauss1Creation_stop;
        cudaEvent_t Gauss2Creation_start;
        cudaEvent_t Gauss2Creation_stop;
        cudaEvent_t DifferenceOfGaussian_start;
        cudaEvent_t DifferenceOfGaussian_stop;
        cudaEvent_t Convolution_start;
        cudaEvent_t Convolution_stop;
    };
    void DogDetect();
    uint8_t* m_src;
    int m_w;
    int m_h;
    int m_gaussKernelSize;
    float m_standardDeviation1;
    float m_standardDeviation2;
    CudaTimers m_timers;
    DogTimings m_timings;
};

#endif //GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOG_EDGE_DETECTION_CUH_
