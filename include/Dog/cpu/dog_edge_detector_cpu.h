//
// Created by Palnit on 2024. 01. 22.
//

#ifndef GPGPU_EDGE_DETECTOR_SRC_DOG_CPU_DOGEDGEDETECTORCPU_H_
#define GPGPU_EDGE_DETECTOR_SRC_DOG_CPU_DOGEDGEDETECTORCPU_H_

#include "include/general/detector_base.h"
#include "include/Dog/dog_timings.h"

class DogEdgeDetectorCPU : public DetectorBase {
public:
    DogEdgeDetectorCPU(SDL_Surface* base, std::string name) : DetectorBase(base,
                                                                           std::move(
                                                                               name)),
                                                              m_w(m_base->w),
                                                              m_h(m_base->h) {}

    void DetectEdge() override;
    void DisplayImGui() override;
    void Display() override;
private:
    int m_w;
    int m_h;
    float* m_pixels1;
    float* m_pixels2;
    float* m_kernel1;
    float* m_kernel2;
    float* m_finalKernel;
    int m_gaussKernelSize = 7;
    float m_standardDeviation1 = 0.1;
    float m_standardDeviation2 = 0.7;
    bool m_timingsReady = false;
    DogTimings m_timings;
};
namespace DetectorsCPU {
void DifferenceOfGaussian(float* kernel1,
                          float* kernel2,
                          float* finalKernel,
                          int kernelSize);
}

#endif //GPGPU_EDGE_DETECTOR_SRC_DOG_CPU_DOGEDGEDETECTORCPU_H_
