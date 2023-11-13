//
// Created by Balint on 2023. 11. 11..
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_MAIN_WINDOW_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_MAIN_WINDOW_H_
#include "include/basic_window.h"

class MainWindow : public BasicWindow {

public:
    MainWindow(const char* title,
               int x,
               int y,
               int w,
               int h,
               Uint32 flags) : BasicWindow(title, x, y, w, h, flags) {
    };
    int Init() override;
    void Render() override;
    void RenderImGui() override;
private:
    GLuint texture = 0;

};

#endif //GPGPU_EDGE_DETECTOR_INCLUDE_MAIN_WINDOW_H_
