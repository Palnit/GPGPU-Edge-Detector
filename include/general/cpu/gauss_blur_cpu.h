//
// Created by Palnit on 2024. 01. 22.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_CPU_GAUSS_BLUR_CPU_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_CPU_GAUSS_BLUR_CPU_H_

#include <cstdint>
#include <functional>
#include <chrono>

namespace DetectorsCPU {
void CopyBack(uint8_t* dest, float* src, int w, int h);
void ConvertGrayScale(uint8_t* base, float* dest, int w, int h);
void GenerateGauss(float* kernel, int kernelSize, float sigma);
void GaussianFilter(float* img,
                    float* dest,
                    float* gauss,
                    int kernelSize,
                    int w,
                    int h);

template<typename F, typename ... Args>
float TimerRunner(F&& func, Args&& ... args) {
    auto t1 = std::chrono::high_resolution_clock::now();
    std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> chrono = t2 - t1;
    return chrono.count();
}
}
#endif //GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_CPU_GAUSS_BLUR_CPU_H_
