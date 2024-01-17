//
// Created by Palnit on 2024. 01. 17.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_OPENGL_SDL_SHADER_PROGRAM_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_OPENGL_SDL_SHADER_PROGRAM_H_

#include "GL/glew.h"
#include <vector>

class ShaderProgram {
public:
    ShaderProgram();
    ~ShaderProgram();
    void AttachShader(GLuint shader);
    void LinkProgram();

    void Bind();
    void UnBind();
private:
    GLuint m_program;
    std::vector<GLuint> m_shaders;
    bool linked = false;
};

#endif //GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_OPENGL_SDL_SHADER_PROGRAM_H_
