//
// Created by Balint on 2023. 11. 11..
//

#include "include/main_window.h"
void MainWindow::Render() {

    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);

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

#if SDL_BYTEORDER == SDL_LIL_ENDIAN
    if (loaded_img->format->BytesPerPixel == 4)
        img_mode = GL_BGRA;
    else
        img_mode = GL_BGR;
#else
    if ( loaded_img->format->BytesPerPixel == 4 )
            img_mode = GL_RGBA;
        else
            img_mode = GL_RGB;
#endif
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
