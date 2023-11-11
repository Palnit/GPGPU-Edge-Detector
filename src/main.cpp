#include <cuda_runtime.h>
#include <cuda.h>

#include "include/gpu_info.h"
#include "include/main_window.h"

/*
    bool quit = false;
    SDL_Event ev;
    SDL_RWops* io = SDL_RWFromFile("img.jpg", "r");
    if (io == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                     "[Load] Error during the SDL initialization: %s",
                     IMG_GetError());
    }
    SDL_Surface* surface = IMG_LoadJPG_RW(io);
    if (surface == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                     "[Surface] Error during the SDL initialization: %s",
                     IMG_GetError);

    }
    std::cout << surface->w << " " << surface->h << std::endl;

    SDL_Renderer* renderer =
        SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surface);
*/

int main(int argc, char* args[]) {
    GetGpuInfo();

    MainWindow win("Edge Detector",
                   SDL_WINDOWPOS_CENTERED,
                   SDL_WINDOWPOS_CENTERED,
                   1024,
                   720,
                   SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN
                       | SDL_WINDOW_RESIZABLE);

    if (win.Init()) {
        return 1;
    }
    return win.run();
}
