//
// Created by Palnit on 2024. 01. 16.
//

#ifndef CUDA_DETECTORCUDA_H_
#define CUDA_DETECTORCUDA_H_

#include "include/general/detector_base.h"
#include "GL/glew.h"
#include "include/general/OpenGL_SDL/element_buffer_object.h"
#include "include/general/OpenGL_SDL/vertex_array_object.h"
#include "include/general/OpenGL_SDL/shader_program.h"

class DetectorCuda : public DetectorBase {
public:
    DetectorCuda(SDL_Surface* base) : DetectorBase(base) {}
    void DetectEdge() override;
    void Display() override;
    void GetTime() override;
private:
    GLuint tex;
    VertexArrayObject VAO;
    VertexBufferObject<float> VBO;
    ElementBufferObject EBO;
    GLuint vertexShader;
    GLuint fragmentShader;
    ShaderProgram shaderProgram;
};

#endif //CUDA_DETECTORCUDA_H_
