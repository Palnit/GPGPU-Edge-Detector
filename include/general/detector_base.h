//
// Created by Palint on 2024. 01. 14.
//

#ifndef DETECTORBASE_H_
#define DETECTORBASE_H_

#include <iostream>
#include <SDL_surface.h>
#include <GL/glew.h>

class DetectorBase {
public:
    DetectorBase(SDL_Surface* picture) : m_base(picture) {}
    virtual void Display() = 0;
    virtual void DetectEdge() = 0;
    virtual void GetTime() = 0;
    static void SetCounter(int counter) {
        m_counter = counter;
    }
protected:
    SDL_Surface* m_base;
    static inline int m_counter;
    int m_position;
    float m_displayVerts[2];
    GLuint VBO;
    GLuint VAO;

};

#endif //DETECTORBASE_H_
