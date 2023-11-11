//
// Created by Balint on 2023. 11. 11..
//

#include "include/basic_window.h"
#include "include/SDL_GLDebugMessageCallback.h"

#include <SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_rwops.h>

#include <GL/glew.h>

#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

inline void HandelSDLError(const char* type) {
    SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                 "[%s] Error during the SDL initialization: %s",
                 type,
                 SDL_GetError());
}

BasicWindow::BasicWindow(const char* title,
                         int x,
                         int y,
                         int w,
                         int h,
                         Uint32 flags) {
    SDL_LogSetPriority(SDL_LOG_CATEGORY_ERROR, SDL_LOG_PRIORITY_ERROR);
    m_title = title;
    m_x = x;
    m_y = y;
    m_w = w;
    m_h = h;
    m_flags = flags;

}
int BasicWindow::run() {
    if (Init()) {
        return 1;
    }
    m_running = true;
    double fpsCount = 0;
    Uint64 fpsLastTime = SDL_GetTicks64();
    ImGuiIO& io = ImGui::GetIO();
    while (m_running) {
        while (SDL_PollEvent(&m_ev)) {
            ImGui_ImplSDL2_ProcessEvent(&m_ev);

            bool isMouseCaptured = io.WantCaptureMouse;
            bool isKeyboardCaptured = io.WantCaptureKeyboard;

            switch (m_ev.type) {
                case SDL_QUIT:
                    m_running = false;
                    break;
                case SDL_KEYDOWN:
                    if (!isKeyboardCaptured)
                        KeyboardDown(m_ev.key);
                    break;
                case SDL_KEYUP:
                    if (!isKeyboardCaptured)
                        KeyboardUp(m_ev.key);
                    break;
                case SDL_MOUSEBUTTONDOWN:
                    if (!isMouseCaptured)
                        MouseDown(m_ev.button);
                    break;
                case SDL_MOUSEBUTTONUP:
                    if (!isMouseCaptured)
                        MouseUp(m_ev.button);
                    break;
                case SDL_MOUSEWHEEL:
                    if (!isMouseCaptured)
                        MouseWheel(m_ev.wheel);
                    break;
                case SDL_MOUSEMOTION:
                    if (!isMouseCaptured)
                        MouseMove(m_ev.motion);
                    break;
                case SDL_WINDOWEVENT:
                    if ((m_ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                        || (m_ev.window.event == SDL_WINDOWEVENT_SHOWN)) {
                        SDL_GetWindowSize(m_window, &m_w, &m_h);
                        Resize();
                    }
                    break;
            }

            Time::DeltaTime = SDL_GetTicks64() - Time::ElapsedTime;
            Time::ElapsedTime = SDL_GetTicks64();
            double diff = (Time::ElapsedTime - fpsLastTime) * 0.001;
            fpsCount++;
            if (diff >= 1.0 / 30.0) {
                Time::FPS = (1.0 / diff) * fpsCount;
                Time::Ms = (diff / fpsCount) * 1000;
                fpsLastTime = Time::ElapsedTime;
                fpsCount = 0;
            }

            Update();
            Render();

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplSDL2_NewFrame();

            ImGui::NewFrame();

            RenderImGui();

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            SDL_GL_SwapWindow(m_window);
        }
    }
    return 0;
}
BasicWindow::~BasicWindow() {
    if (m_window != nullptr) {
        SDL_DestroyWindow(m_window);
    }
    if (m_context != nullptr) {
        SDL_GL_DeleteContext(m_context);
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    SDL_Quit();
}
int BasicWindow::Init() {

    if (SDL_Init(SDL_INIT_VIDEO) == -1) {
        HandelSDLError("SDL initialization");
        return 1;
    }

    if (IMG_Init(IMG_INIT_JPG) == 0) {
        HandelSDLError("SDL IMG initialization");
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_CORE);

#ifdef _DEBUG
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
#endif

    SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    m_window = SDL_CreateWindow(m_title, m_x, m_y, m_w, m_h, m_flags);

    if (m_window == nullptr) {
        HandelSDLError("Window initialization");
        return 1;
    }

    m_context = SDL_GL_CreateContext(m_window);

    if (m_context == nullptr) {
        HandelSDLError("GL Context initialization");
        return 1;
    }

    SDL_GL_MakeCurrent(m_window, m_context);
    SDL_GL_SetSwapInterval(1);

    GLenum err = glewInit();
    if (err != GLEW_OK) {
        SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                     "[GLEW] Error during the initialization of glew.");
        return 1;
    }

    GLint context_flags;
    glGetIntegerv(GL_CONTEXT_FLAGS, &context_flags);
    if (context_flags & GL_CONTEXT_FLAG_DEBUG_BIT) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageControl(GL_DONT_CARE,
                              GL_DONT_CARE,
                              GL_DEBUG_SEVERITY_NOTIFICATION,
                              0,
                              nullptr,
                              GL_FALSE);
        glDebugMessageCallback(SDL_GLDebugMessageCallback, nullptr);
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |=
        ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForOpenGL(m_window, m_context);
    ImGui_ImplOpenGL3_Init();

    return 0;
}
