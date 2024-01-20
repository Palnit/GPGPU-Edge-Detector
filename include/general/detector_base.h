//
// Created by Palint on 2024. 01. 14.
//

#ifndef DETECTORBASE_H_
#define DETECTORBASE_H_

#include <iostream>
#include <utility>
#include <SDL_surface.h>
#include <GL/glew.h>
#include "include/general/OpenGL_SDL/vertex_array_object.h"
#include "include/general/OpenGL_SDL/shader_program.h"

class DetectorBase {
public:
    DetectorBase(SDL_Surface* picture, std::string name)
        : m_base(picture), m_name(std::move(name)) {
        m_detected = SDL_CreateRGBSurface(0,
                                          m_base->w,
                                          m_base->h,
                                          m_base->format->BitsPerPixel,
                                          m_base->format->Rmask,
                                          m_base->format->Gmask,
                                          m_base->format->Bmask,
                                          m_base->format->Amask);
        SDL_BlitSurface(m_base, NULL, m_detected, NULL);
    }
    ~DetectorBase() {
        SDL_FreeSurface(m_base);
        SDL_FreeSurface(m_detected);
    }
    virtual void Display() = 0;
    virtual void DetectEdge() = 0;
    virtual void DisplayImGui() = 0;
    static void SetCounter(int counter) {
        m_counter = counter;
    }
protected:
    SDL_Surface* m_base;
    SDL_Surface* m_detected;
    static inline int m_counter;
    int m_position;
    std::string m_name;
    GLuint tex;
    VertexArrayObject VAO;
    VertexBufferObject<float> VBO;
    ElementBufferObject EBO;
    GLuint vertexShader;
    GLuint fragmentShader;
    ShaderProgram shaderProgram;

};

#endif //DETECTORBASE_H_
