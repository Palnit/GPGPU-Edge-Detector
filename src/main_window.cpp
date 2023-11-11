//
// Created by Balint on 2023. 11. 11..
//

#include "include/main_window.h"
#include <implot.h>

void MainWindow::Render() {

    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glViewport(0, 0, m_w, m_h);
    glCullFace(GL_BACK);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, test);
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
int MainWindow::Init() {
    if (BasicWindow::Init()) {
        return 1;
    }
    SDL_Surface* loaded_img = IMG_Load("img.jpg");

    printf("b/p %d\n", loaded_img->format->BytesPerPixel);

    int bpp = loaded_img->format->BytesPerPixel;

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    Uint32 format = SDL_PIXELFORMAT_ABGR8888;
#else
    Uint32 format = SDL_PIXELFORMAT_RGBA8888;
#endif

    SDL_Surface* nloaded_img = SDL_ConvertSurfaceFormat(loaded_img, format, 0);
    printf("test %d\n", nloaded_img->format->BytesPerPixel);

    uint8_t red, green, blue, alpha;
    auto a = SDL_GetTicks64();

    auto b = SDL_GetTicks64();
    printf("Time: %f\n", (b - a) * 0.0001);
    for (int i = 0; i < (nloaded_img->w - 1); ++i) {
        for (int j = 0; j < (nloaded_img->h - 1); ++j) {
            Uint32* pixel =
                (Uint32*) ((Uint8*) nloaded_img->pixels
                    + i * nloaded_img->format->BytesPerPixel
                    + j * nloaded_img->pitch);

            Color color = {*pixel};

            Uint8 grey = 0.299 * color.channels.r
                + 0.587 * color.channels.g
                + 0.114 * color.channels.b;

            Color greyscale = {0};
            greyscale.channels = {grey, grey, grey, color.channels.a};
            *pixel = greyscale.raw;
        }
    }
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
    test = tex;

    return 0;
}
void MainWindow::RenderImGui() {
    bool t = true;
    ImGui::ShowMetricsWindow(&t);

}
