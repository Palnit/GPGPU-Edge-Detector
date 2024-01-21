//
// Created by Palnit on 2023. 11. 11.
//

#include "include/general//main_window.h"
#include "include/general/detector_base.h"
#include "include/general/OpenGL_SDL/file_handling.h"
#include "include/general/OpenGL_SDL/generic_structs.h"

#include <ctime>

#include <cuda_runtime.h>

#include <implot.h>

int MainWindow::Init() {

    uint8_t grey;
    RGBA* color;
    /*for (int i = 0; i < (nloaded_img->w - 1); ++i) {
        for (int j = 0; j < (nloaded_img->h - 1); ++j) {
            color =
                (RGBA*) (Uint32*) ((Uint8*) nloaded_img->pixels
                    + i * nloaded_img->format->BytesPerPixel
                    + j * nloaded_img->pitch);

            if (i < 10 & j < 10) {

                std::printf("cpu : %d,%d,%d,%d,%d\n",
                            i * j,
                            color->r,
                            color->g,
                            color->b,
                            color->a);
            }

            /*alpha = (*pixel & 0xFF000000) >> 24;
            blue = (*pixel & 0x00FF0000) >> 16;
            green = (*pixel & 0x0000FF00) >> 8;
            red = (*pixel & 0x000000FF);

            grey = (0.299 * red) + (0.587 * green) + (0.114 * blue);
            *pixel = (alpha << 24) | (grey << 16) | (grey << 8) | grey;*/

    /*color->r = color->g = color->b =
        0.299 * color->r
            + 0.587 * color->g
            + 0.114 * color->b;
}
}*/

    return 0;
}

void MainWindow::Render() {
    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glViewport(0, 0, m_width, m_height);
    glCullFace(GL_BACK);
    glClear(GL_COLOR_BUFFER_BIT);
    for (DetectorBase* detector : m_detectors) {
        detector->Display();
    }
}

void MainWindow::RenderImGui() {
    bool t = true;
    ImGui::ShowMetricsWindow(&t);
    ImGui::ShowDemoWindow(&t);
    m_display.DisplayImGui();

}
MainWindow::~MainWindow() {
    for (DetectorBase* detector : m_detectors) {
        free(detector);
    }
}
void MainWindow::AddDetector(DetectorBase* Detector) {
    m_detectors.push_back(Detector);
}
void MainWindow::RemoveDetector(DetectorBase* Detector) {
    auto position = std::find(m_detectors.begin(), m_detectors.end(), Detector);
    if (position != m_detectors.end()) {
        free(*position);
        m_detectors.erase(position);
    }
}
void MainWindow::Resize() {
    BasicWindow::Resize();
    m_display.Resize(m_width, m_height);
}
