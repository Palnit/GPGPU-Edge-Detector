//
// Created by Palnit on 2024. 01. 21.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOGEDGEDETECTORCUDA_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOGEDGEDETECTORCUDA_H_

#include "include/general/detector_base.h"
#include "include/Dog/dog_timings.h"

class DogEdgeDetectorCuda : public DetectorBase {
public:
    DogEdgeDetectorCuda(SDL_Surface* base,
                        std::string name) : DetectorBase(
        base,
        std::move(name)) {
    }
    void DetectEdge() override;
    void DisplayImGui() override;
    void Display() override;
private:
    int m_gaussKernelSize = 17;
    float m_standardDeviation1 = 0.1;
    float m_standardDeviation2 = 10;
    bool m_timingsReady = false;
    DogTimings m_timings;
};

#endif //GPGPU_EDGE_DETECTOR_INCLUDE_CUDA_DOGEDGEDETECTORCUDA_H_
