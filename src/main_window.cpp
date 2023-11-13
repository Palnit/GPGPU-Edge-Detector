//
// Created by Balint on 2023. 11. 11..
//

#include "include/main_window.h"
#include <implot.h>
#include "cuda_include/cuda_edge_detection.cuh"
#include <cuda_runtime.h>

int MainWindow::Init() {
    if (BasicWindow::Init()) {
        return 1;
    }
    SDL_Surface* loaded_img = IMG_Load("pictures/img.jpg");

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    Uint32 format = SDL_PIXELFORMAT_ABGR8888;
#else
    Uint32 format = SDL_PIXELFORMAT_RGBA8888;
#endif

    SDL_Surface* nloaded_img = SDL_ConvertSurfaceFormat(loaded_img, format, 0);
    auto a = SDL_GetTicks64();
    uint8_t* h_pixel = 0;
    cudaError asd2 = cudaMalloc((void**) h_pixel,
                                1);
    if (asd2 == cudaErrorInvalidValue) {
        printf("yay");
    }
    cudaMemcpy(h_pixel,
               nloaded_img->pixels,
               1,
               cudaMemcpyHostToDevice);
    test(1, 10, h_pixel);
    cudaFree(h_pixel);
    uint8_t grey;
    RGBA* color;
    for (int i = 0; i < (nloaded_img->w - 1); ++i) {
        for (int j = 0; j < (nloaded_img->h - 1); ++j) {
            color =
                (RGBA*) (Uint32*) ((Uint8*) nloaded_img->pixels
                    + i * nloaded_img->format->BytesPerPixel
                    + j * nloaded_img->pitch);

            /*alpha = (*pixel & 0xFF000000) >> 24;
            blue = (*pixel & 0x00FF0000) >> 16;
            green = (*pixel & 0x0000FF00) >> 8;
            red = (*pixel & 0x000000FF);

            grey = (0.299 * red) + (0.587 * green) + (0.114 * blue);
            *pixel = (alpha << 24) | (grey << 16) | (grey << 8) | grey;*/

            color->r = color->g = color->b =
                0.299 * color->r
                    + 0.587 * color->g
                    + 0.114 * color->b;
        }
    }
    auto b = SDL_GetTicks64();
    printf("Time: %f\n", (b - a) * 0.0001);
    GLuint tex;
    glGenTextures(1, &tex);

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 nloaded_img->w,
                 nloaded_img->h,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 nloaded_img->pixels);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    SDL_FreeSurface(nloaded_img);
    texture = tex;

    return 0;
}

void MainWindow::Render() {

    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glViewport(0, 0, m_width, m_height);
    glCullFace(GL_BACK);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_TRIANGLES);

    glTexCoord2i(0, 1);
    glVertex2i(-1, -1);

    glTexCoord2i(0, 0);
    glVertex2i(-1, 1);

    glTexCoord2i(1, 0);
    glVertex2i(1, 1);

    glTexCoord2i(1, 0);
    glVertex2i(1, 1);

    glTexCoord2i(1, 1);
    glVertex2i(1, -1);

    glTexCoord2i(0, 1);
    glVertex2i(-1, -1);

    glEnd();

}

void MainWindow::RenderImGui() {
    bool t = true;
    ImGui::ShowMetricsWindow(&t);

}
