//
// Created by Palnit on 2024. 01. 17.
//

#ifndef GPGPU_EDGE_DETECTOR_SRC_GENERAL_OPENGL_SDL_ELEMENTBUFFEROBJECT_H_
#define GPGPU_EDGE_DETECTOR_SRC_GENERAL_OPENGL_SDL_ELEMENTBUFFEROBJECT_H_

#include <vector>
#include "GL/glew.h"

class ElementBufferObject {
public:
    ElementBufferObject();
    ElementBufferObject(GLenum usage);
    template<auto size>
    ElementBufferObject(unsigned int (& elements)[size],
                        GLenum usage) : m_usage(usage) {
        glGenBuffers(1, &m_EBO);
        m_elements.insert(m_elements.end(),
                          elements,
                          elements + size);
    }
    ElementBufferObject(std::initializer_list<unsigned int> list,
                        GLenum usage) : m_usage(usage) {
        glGenBuffers(1, &m_EBO);
        m_elements.insert(m_elements.end(),
                          list.begin(),
                          list.end());
    }
    ~ElementBufferObject();
    void SetUsage(GLenum usage);
    void AddElement(unsigned int element);
    template<auto size>
    void AddElement(unsigned int (& elements)[size]) {
        m_elements.insert(m_elements.end(),
                          elements,
                          elements + size);
        m_set = false;
    }
    void AddElement(std::initializer_list<unsigned int> list);
    void Bind();
    static void UnBind();

private:
    std::vector<unsigned int> m_elements;
    GLuint m_EBO;
    GLenum m_usage;
    bool m_set = false;
};

#endif //GPGPU_EDGE_DETECTOR_SRC_GENERAL_OPENGL_SDL_ELEMENTBUFFEROBJECT_H_
