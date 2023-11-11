//
// Created by Balint on 2023. 11. 11..
//

#ifndef BASIC_WINDOW_H_
#define BASIC_WINDOW_H_

#include <SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_rwops.h>

#include <GL/glew.h>

#include <glm/glm.hpp>

#include <imgui.h>

inline void HandelSDLError(const char* type);

class Time {
public:
    static inline Uint64 ElapsedTime = 0;
    static inline Uint64 DeltaTime = 0;
};

class BasicWindow {

public:
    BasicWindow(const char* title, int x, int y, int w, int h, Uint32 flags);
    virtual void KeyboardDown(const SDL_KeyboardEvent&) {};
    virtual void KeyboardUp(const SDL_KeyboardEvent&) {};
    virtual void MouseDown(const SDL_MouseButtonEvent&) {};
    virtual void MouseUp(const SDL_MouseButtonEvent&) {};
    virtual void MouseWheel(const SDL_MouseWheelEvent&) {};
    virtual void MouseMove(const SDL_MouseMotionEvent&) {};
    virtual int Init();
    virtual void Resize() {};
    virtual void Update() {};
    virtual void Render() {
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
    };
    virtual void RenderImGui() {
        bool render = true;
        ImGui::ShowDemoWindow(&render);
    };

    ~BasicWindow();
    int run();

protected:
    const char* m_title;
    int m_x;
    int m_y;
    int m_w;
    int m_h;
    bool m_running;
    Uint32 m_flags;
    SDL_Window* m_window = nullptr;
    SDL_GLContext m_context;
    SDL_Event m_ev;
};

#endif //BASIC_WINDOW_H_
