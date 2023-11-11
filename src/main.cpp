#include <cuda_runtime.h>
#include <cuda.h>

#include <SDL.h>

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
    while (!quit) {
        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
                case SDL_QUIT:quit = true;
                    break;
            }
        }
    }
    SDL_DestroyWindow(win);
    return 0;
}

int main(int argc, char* args[]) {
    GetGpuInfo();

    return CreatWindow();
}
