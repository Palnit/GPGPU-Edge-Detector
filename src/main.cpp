#include <cuda_runtime.h>
#include <cuda.h>

#include <SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_rwops.h>

#include <iostream>

#include "include/gpu_info.h"
void CleanUp() { SDL_Quit(); }

int CreatWindow() {
    SDL_LogSetPriority(SDL_LOG_CATEGORY_ERROR, SDL_LOG_PRIORITY_ERROR);

    if (SDL_Init(SDL_INIT_VIDEO) == -1) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                     "[SDL initialization] Error during the SDL initialization: %s",
                     SDL_GetError());
        return 1;
    }

    if (IMG_Init(IMG_INIT_JPG) == 0) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                     "[Image] Error during the SDL initialization: %s",
                     SDL_GetError());
        return 1;
    }

    std::atexit(CleanUp);
    SDL_Window* win = nullptr;
    win = SDL_CreateWindow("Edge Detector",
                           100,
                           100,
                           800,
                           600,
                           SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (win == nullptr) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                     "[Window creation] Error during the SDL initialization: %s",
                     SDL_GetError());
        return 1;
    }
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
                     IMG_GetError());

    }
    std::cout << surface->w << " " << surface->h << std::endl;

    SDL_Renderer* renderer =
        SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surface);

    while (!quit) {
        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
                case SDL_QUIT:quit = true;
                    break;
            }
        }
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, tex, NULL, NULL);
        SDL_RenderPresent(renderer);
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(win);
    return 0;
}

int main(int argc, char* args[]) {
    GetGpuInfo();

    return CreatWindow();
}
