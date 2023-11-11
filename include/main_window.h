//
// Created by Balint on 2023. 11. 11..
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_MAIN_WINDOW_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_MAIN_WINDOW_H_
#include "include/basic_window.h"

typedef struct RGBA {
    Uint8 r;
    Uint8 g;
    Uint8 b;
    Uint8 a;
} RGBA;

typedef union Color {
    Uint32 raw;
    RGBA channels;
} Color;

class MainWindow : public BasicWindow {

public:
    MainWindow(const char* title,
               int x,
               int y,
               int w,
               int h,
               Uint32 flags) : BasicWindow(title, x, y, w, h, flags) {
    };
    void Render() override;
    int Init() override;
    void RenderImGui() override;
private:
    GLuint test = 0;

};

#endif //GPGPU_EDGE_DETECTOR_INCLUDE_MAIN_WINDOW_H_
