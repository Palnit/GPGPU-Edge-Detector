//
// Created by Palnit on 2024. 01. 21.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_CANNY_CPU_CANNY_EDGE_DETECTOR_CPU_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_CANNY_CPU_CANNY_EDGE_DETECTOR_CPU_H_

#include "include/general/detector_base.h"
#include "include/Canny/canny_timings.h"

class CannyEdgeDetectorCPU : public DetectorBase {
public:
    CannyEdgeDetectorCPU(SDL_Surface* base,
                         std::string name) : DetectorBase(base,
                                                          std::move(name)),
                                             m_w(m_base->w),
                                             m_h(m_base->h) {

    }
    void DetectEdge() override;
    void DisplayImGui() override;
    void Display() override;
private:

    int m_w;
    int m_h;
    int m_gaussKernelSize = 3;
    float m_standardDeviation = 1;
    float m_highTrashHold = 150;
    float m_lowTrashHold = 100;
    bool m_timingsReady = false;
    float* m_pixels1;
    float* m_pixels2;
    float* m_kernel;
    float* m_tangent;
    CannyTimings m_timings;
};
namespace DetectorsCPU {
void DetectionOperator(float* src, float* dest, float* tangent, int w, int h);
void NonMaximumSuppression(float* src,
                           float* dest,
                           float* tangent,
                           int w,
                           int h);
void DoubleThreshold(float* src,
                     float* dest,
                     int w,
                     int h,
                     float high,
                     float low);
void Hysteresis(float* src, float* dest, int w, int h);
}
#endif //GPGPU_EDGE_DETECTOR_INCLUDE_CANNY_CPU_CANNY_EDGE_DETECTOR_CPU_H_
