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

    glTexCoord2i(1, 1);
    glVertex2i(-1, -1);

    glTexCoord2i(1, 0);
    glVertex2i(-1, 1);

    glTexCoord2i(0, 0);
    glVertex2i(1, 1);

    glTexCoord2i(0, 0);
    glVertex2i(1, 1);

    glTexCoord2i(0, 1);
    glVertex2i(1, -1);

    glTexCoord2i(1, 1);
    glVertex2i(-1, -1);

    glEnd();

}
int MainWindow::Init() {
    if (BasicWindow::Init()) {
        return 1;
    }
    SDL_Surface* loaded_img = IMG_Load("img.jpg");
    int img_mode = 0;

    if (loaded_img->format->BytesPerPixel == 4)
        img_mode = GL_RGBA;
    else
        img_mode = GL_RGB;

    printf("b/p %d", loaded_img->format->BytesPerPixel);

    int bpp = loaded_img->format->BytesPerPixel;
    SDL_PixelFormat* fmt;
    fmt = loaded_img->format;
    /* Here p is the address to the pixel we want to retrieve */
    uint8_t* pixel;
    uint8_t red, green, blue;
    auto a = SDL_GetTicks64();
    for (int i = 0; i < (loaded_img->h - 1); ++i) {
        for (int j = 0; j < (loaded_img->w - 1); ++j) {

            pixel = (Uint8*) loaded_img->pixels + i * loaded_img->pitch
                + j * bpp;

            red = pixel[0];
            green = pixel[1];
            blue = pixel[2];

            pixel[0] = pixel[1] = pixel[2] = (red + green + blue) / 3;

        }
    }

    auto b = SDL_GetTicks64();
    printf("Time: %f", (b - a) * 0.0001);

    GLuint tex;
    glGenTextures(1, &tex);

    glBindTexture(GL_TEXTURE_2D, tex);
    gluBuild2DMipmaps(GL_TEXTURE_2D,
                      GL_RGBA,
                      loaded_img->w,
                      loaded_img->h,
                      img_mode,
                      GL_UNSIGNED_BYTE,
                      loaded_img->pixels);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    SDL_FreeSurface(loaded_img);
    test = tex;

    return 0;
}
void MainWindow::RenderImGui() {
    bool t = true;
    ImGui::ShowMetricsWindow(&t);

}
