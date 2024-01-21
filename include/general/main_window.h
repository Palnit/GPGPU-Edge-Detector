//
// Created by Palnit on 2023. 11. 11.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_MAIN_WINDOW_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_MAIN_WINDOW_H_
#include "include/general/OpenGL_SDL/basic_window.h"
#include "include/general/detector_base.h"
#include "include/Canny/cuda/canny_edge_detector_cuda.h"
#include "include/general/OpenGL_SDL/file_handling.h"
#include "imgui_display.h"

class MainWindow : public BasicWindow {

public:
    MainWindow::MainWindow(const char* title,
                           int x,
                           int y,
                           int w,
                           int h,
                           Uint32 flags) : BasicWindow(title,
                                                       x,
                                                       y,
                                                       w,
                                                       h,
                                                       flags),
                                           m_display(m_width, m_height, this) {
    }
    ~MainWindow();
    int Init() override;
    void Render() override;
    void RenderImGui() override;
    void AddDetector(DetectorBase* Detector);
    void RemoveDetector(DetectorBase* Detector);
    void Resize() override;
private:
    std::vector<DetectorBase*> m_detectors;
    CannyEdgeDetectorCuda* m_det;
    ImGuiDisplay m_display;

};

#endif //GPGPU_EDGE_DETECTOR_INCLUDE_MAIN_WINDOW_H_
