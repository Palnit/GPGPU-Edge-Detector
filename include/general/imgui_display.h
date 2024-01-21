//
// Created by Palnit on 2024. 01. 20.
//

#ifndef GPGPU_EDGE_DETECTOR_SRC_GENERAL_IMGUI_DISPLAY_H_
#define GPGPU_EDGE_DETECTOR_SRC_GENERAL_IMGUI_DISPLAY_H_

#include <vector>
#include "detector_base.h"
#include "include/general/OpenGL_SDL/basic_window.h"

class ImGuiDisplay {
public:
    ImGuiDisplay(int width, int height, BasicWindow* parent)
        : m_width(width), m_height(height), m_parent(parent) {
    }
    void DisplayImGui();
    static bool VectorOfStringGetter(void* data, int n, const char** out_text) {
        const std::vector<std::string>* v = (std::vector<std::string>*) data;
        *out_text = v->at(n).c_str();
        return true;
    }
    void Resize(int width, int height) {
        m_width = width;
        m_height = height;
    }
private:
    std::vector<DetectorBase*> m_detectors;
    std::vector<std::string> m_names;
    BasicWindow* m_parent;
    int m_width;
    int m_height;
    int m_add = 0;
    int m_remove = 0;
    int m_picture = 0;
    char m_buf[300] = "Name Detector";
    std::vector<std::string>
        m_pictures{"d2.png", "d2_90.jpg", "img.jpg", "test.jpg", "github.PNG"};
};

#endif //GPGPU_EDGE_DETECTOR_SRC_GENERAL_IMGUI_DISPLAY_H_
