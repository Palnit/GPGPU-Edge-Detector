//
// Created by Palnit on 2024. 01. 17.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_OPENGL_SDL_VERTEXBUFFEROBJECT_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_OPENGL_SDL_VERTEXBUFFEROBJECT_H_

#include <vector>
#include "GL/glew.h"

template<typename T>
class VertexBufferObject {
public:
    class AttributeDescriptor;

    VertexBufferObject() : m_usage(GL_STATIC_DRAW) {
        glGenBuffers(1, &m_VBO);
    }
    VertexBufferObject(GLenum usage) : m_usage(usage) {
        glGenBuffers(1, &m_VBO);
    };
    template<auto size>
    VertexBufferObject(T (& elements)[size],
                       GLenum usage) : m_usage(usage) {
        glGenBuffers(1, &m_VBO);
        m_elements.insert(m_elements.end(),
                          elements,
                          elements + size);
    }
    VertexBufferObject(std::initializer_list<T> list,
                       GLenum usage) : m_usage(usage) {
        glGenBuffers(1, &m_VBO);
        m_elements.insert(m_elements.end(),
                          list.begin(),
                          list.end());
    }
    ~VertexBufferObject() {
        m_elements.clear();
        m_desc.clear();
        glDeleteBuffers(1, &m_VBO);
    }
    void SetUsage(GLenum usage) {
        m_usage = usage;
    }
    void AddElement(T element) {
        m_elements.push_back(element);
        m_set = false;
    }
    template<auto size>
    void AddElement(T (& elements)[size]) {
        m_elements.insert(m_elements.end(),
                          elements,
                          elements + size);
        m_set = false;
    }
    void AddElement(std::initializer_list<T> list) {
        m_elements.insert(m_elements.end(), list.begin(), list.end());
        m_set = false;
    };

    void AddAttribute(std::initializer_list<AttributeDescriptor> list) {
        m_desc.insert(m_desc.end(), list.begin(), list.end());
    }
    void AddAttribute(AttributeDescriptor element) {
        m_desc.push_back(element);
    }

    void Bind() {
        glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
        if (!m_set) {
            glBufferData(GL_ARRAY_BUFFER,
                         sizeof(T) * m_elements.size(),
                         m_elements.data(),
                         m_usage);
            m_set = true;
        }

    }
    static void UnBind() {
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    struct AttributeDescriptor {
        AttributeDescriptor(GLint size, GLsizei stride,
                            GLenum type,
                            GLboolean normalized,
                            const GLvoid* offset)
            : size(size),
              stride(stride),
              type(type),
              normalized(normalized),
              offset(offset) {
        }
        AttributeDescriptor(GLint size,
                            GLsizei stride,
                            GLenum type,
                            const GLvoid* offset)
            : size(size),
              stride(stride),
              type(type),
              normalized(GL_FALSE),
              offset(offset) {

        }
        AttributeDescriptor(GLint size, GLsizei stride, const GLvoid* offset)
            : size(size),
              stride(stride),
              type(GL_FLOAT),
              normalized(GL_FALSE),
              offset(offset) {
        }
        GLint size;
        GLenum type;
        GLboolean normalized;
        GLsizei stride;
        const GLvoid* offset;
    };
    const std::vector<AttributeDescriptor>& GetDescriptors() {
        return m_desc;
    }
private:
    std::vector<T> m_elements;
    GLuint m_VBO;
    GLenum m_usage;
    bool m_set = false;
    std::vector<AttributeDescriptor> m_desc;
};

#endif //GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_OPENGL_SDL_VERTEXBUFFEROBJECT_H_
