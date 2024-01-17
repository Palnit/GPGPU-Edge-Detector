//
// Created by Palnit on 2024. 01. 16.
//

#include "include/cuda/detector_cuda.h"
#include "include/cuda/cuda_edge_detection.cuh"
#include "SDL_timer.h"
#include "include/general/OpenGL_SDL/basic_window.h"
#include "include/general/OpenGL_SDL/file_handling.h"

void DetectorCuda::DetectEdge() {
    uint8_t* h_pixel = nullptr;

    cudaMalloc((void**) &h_pixel,
               sizeof(uint8_t) * m_base->w
                   * m_base->h
                   * m_base->format->BytesPerPixel);

    cudaMemcpy(h_pixel,
               m_base->pixels,
               sizeof(uint8_t) * m_base->w * m_base->h
                   * m_base->format->BytesPerPixel,
               cudaMemcpyHostToDevice);

    dim3 threads(32, 32);
    dim3 block
        (m_base->w / threads.x + (m_base->w % threads.x == 0 ? 0 : 1),
         m_base->h / threads.y
             + (m_base->h % threads.y == 0 ? 0 : 1));

    auto a = SDL_GetTicks64();
    test(block, threads, h_pixel, m_base->w, m_base->h);
    auto b = SDL_GetTicks64();
    printf("Time: %f\n", (b - a) * 0.0001);
    cudaMemcpy(m_base->pixels,
               h_pixel,
               sizeof(uint8_t) * m_base->w * m_base->h
                   * m_base->format->BytesPerPixel,
               cudaMemcpyDeviceToHost);

    cudaFree(h_pixel);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 m_base->w,
                 m_base->h,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 m_base->pixels);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    SDL_FreeSurface(m_base);

    float verts[] = {
        -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
        1.0f, -1.0f, 0.0f, 1.0f, 1.0f
    };

    VBO.AddElement(verts);
    EBO.AddElement({2, 3, 1, 3, 0, 1});

    vertexShader = FileHandling::LoadShader(GL_VERTEX_SHADER,
                                            "shaders/default_vertex.vert");

    fragmentShader = FileHandling::LoadShader(GL_FRAGMENT_SHADER,
                                              "shaders/default_fragment.frag");

    shaderProgram.AttachShader(vertexShader);
    shaderProgram.AttachShader(fragmentShader);

    VBO.AddAttribute({{3, 5 * sizeof(float), (void*) 0},
                      {2, 5 * sizeof(float), (void*) (3 * sizeof(float))}});
    VAO.AddVertexBuffer(VBO);
    VAO.AddElementBuffer(EBO);
}
void DetectorCuda::Display() {
    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glViewport(0, 0, 1024, 720);
    glCullFace(GL_BACK);
    glClear(GL_COLOR_BUFFER_BIT);
    shaderProgram.Bind();
    VAO.Bind();
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    VAO.UnBind();
    shaderProgram.UnBind();
}
void DetectorCuda::GetTime() {

}
