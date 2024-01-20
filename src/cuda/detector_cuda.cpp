//
// Created by Palnit on 2024. 01. 16.
//

#include "include/cuda/detector_cuda.h"
#include "include/cuda/cuda_edge_detection.cuh"
#include "SDL_timer.h"
#include "include/general/OpenGL_SDL/basic_window.h"
#include "include/general/OpenGL_SDL/file_handling.h"

#include <cuda_runtime.h>

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

    auto a = SDL_GetTicks64();
    CudaDetector(h_pixel,
                 m_base->w,
                 m_base->h,
                 m_gaussKernelSize,
                 m_standardDeviation,
                 m_highTrashHold,
                 m_lowTrashHold);
    auto b = SDL_GetTicks64();
    printf("Time: %f\n", (b - a) * 0.0001);
    cudaMemcpy(m_detected->pixels,
               h_pixel,
               sizeof(uint8_t) * m_detected->w * m_detected->h
                   * m_detected->format->BytesPerPixel,
               cudaMemcpyDeviceToHost);

    cudaFree(h_pixel);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 m_detected->w,
                 m_detected->h,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 m_detected->pixels);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

}
void DetectorCuda::Display() {
    shaderProgram.Bind();
    VAO.Bind();
    glBindTexture(GL_TEXTURE_2D, tex);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    VAO.UnBind();
    shaderProgram.UnBind();
}
DetectorCuda::DetectorCuda(SDL_Surface* base, std::string name) : DetectorBase(
    base,
    std::move(name)) {
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 m_detected->w,
                 m_detected->h,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 m_detected->pixels);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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
void DetectorCuda::DisplayImGui() {
    if (ImGui::BeginTabItem(m_name.c_str())) {

        if (ImGui::SliderInt("Gauss Kernel Size", &m_gaussKernelSize, 3, 7)) {
            if (m_gaussKernelSize % 2 == 0) {
                m_gaussKernelSize++;
            }
        }
        ImGui::SetItemTooltip("Only Odd Numbers");
        ImGui::SliderFloat("Standard Deviation",
                           &m_standardDeviation,
                           1.0f,
                           10.0f);
        ImGui::SliderFloat("High Trash Hold",
                           &m_highTrashHold,
                           1.0f,
                           255.0f);
        ImGui::SliderFloat("Low Trash Hold",
                           &m_lowTrashHold,
                           1.0f,
                           255.0f);
        ImGui::Separator();
        if (ImGui::Button("Detect")) {
            DetectEdge();
        }
        ImGui::EndTabItem();
    }
}