//
// Created by Palnit on 2024. 01. 16.
//

#ifndef CUDA_DETECTORCUDA_H_
#define CUDA_DETECTORCUDA_H_

#include <utility>

#include "include/general/detector_base.h"
#include "GL/glew.h"
#include "include/general/OpenGL_SDL/element_buffer_object.h"
#include "include/general/OpenGL_SDL/vertex_array_object.h"
#include "include/general/OpenGL_SDL/shader_program.h"

class DetectorCuda : public DetectorBase {
public:
    DetectorCuda(SDL_Surface* base, std::string name);
    void DetectEdge() override;
    void DisplayImGui() override;
    void Display() override;
private:
    int m_gaussKernelSize = 3;
    float m_standardDeviation = 1;
    float m_highTrashHold = 150;
    float m_lowTrashHold = 100;
};

#endif //CUDA_DETECTORCUDA_H_
