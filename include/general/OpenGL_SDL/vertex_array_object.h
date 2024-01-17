//
// Created by Palnit on 2024. 01. 17.
//

#ifndef GPGPU_EDGE_DETECTOR_SRC_GENERAL_OPENGL_SDL_VERTEXARRAYOBJECT_H_
#define GPGPU_EDGE_DETECTOR_SRC_GENERAL_OPENGL_SDL_VERTEXARRAYOBJECT_H_

#include "GL/glew.h"
#include "vertex_buffer_object.h"
#include "element_buffer_object.h"
class VertexArrayObject {
public:
    VertexArrayObject() : m_count(0) {
        glGenVertexArrays(1, &m_VAO);
    }

    void Bind() const;
    void UnBind() const;

    template<typename T>
    void AddVertexBuffer(VertexBufferObject<T> VBO) {
        Bind();
        VBO.Bind();
        for (unsigned int i = 0; i < VBO.GetDescriptors().size(); i++) {
            glVertexAttribPointer(m_count,
                                  VBO.GetDescriptors()[i].size,
                                  VBO.GetDescriptors()[i].type,
                                  VBO.GetDescriptors()[i].normalized,
                                  VBO.GetDescriptors()[i].stride,
                                  VBO.GetDescriptors()[i].offset);
            glEnableVertexAttribArray(m_count);
            m_count++;
        }
        VBO.UnBind();
        UnBind();
    }
    void AddElementBuffer(ElementBufferObject EBO) {
        Bind();
        EBO.Bind();
        UnBind();
    }

private:
    GLuint m_VAO;
    GLuint m_count;

};

#endif //GPGPU_EDGE_DETECTOR_SRC_GENERAL_OPENGL_SDL_VERTEXARRAYOBJECT_H_
