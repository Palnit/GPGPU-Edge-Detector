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
#include <implot.h>
#include <vector>

inline void HandelSDLError(const char* type);

typedef struct RGBA {
    Uint8 r;
    Uint8 g;
    Uint8 b;
    Uint8 a;
} RGBA;

typedef union Color {
    Uint32 raw;
    RGBA channels;
} Color;

class Time {
public:
    static inline Uint64 ElapsedTime = 0;
    static inline Uint64 DeltaTime = 0;
    static inline double FPS = 0;
    static inline double Ms = 0;
};

/*!
 * \class BasicWindow
 * \brief An SDL2-Opengl-ImGui window generating virtual class should be inherited
 * Creates and checks for errors during window and opengl context creation its a
 * its a class that's designed to be inherited from to completely abstract the sdl2
 * - opengl - ImGui window creation boiler plate and provide easy access to it's
 * variables while having the convenience to not pollute your code with
 * having to implement all keyboard functions even if you dont use them
 */
class BasicWindow {

public:
    /*!
     * Constructor
     * \param title The title of the window
     * \param x
     * \param y
     * \param width
     * \param height
     * \param flags
     */

    BasicWindow(const char* title,
                int x,
                int y,
                int width,
                int height,
                Uint32 flags);
    virtual void KeyboardDown(const SDL_KeyboardEvent& ev) {};
    virtual void KeyboardUp(const SDL_KeyboardEvent& ev) {};
    virtual void MouseDown(const SDL_MouseButtonEvent& ev) {};
    virtual void MouseUp(const SDL_MouseButtonEvent& ev) {};
    virtual void MouseWheel(const SDL_MouseWheelEvent& ev) {};
    virtual void MouseMove(const SDL_MouseMotionEvent& ev) {};
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
    int m_width;
    int m_height;
    bool m_running;
    Uint32 m_flags;
    SDL_Window* m_window = nullptr;
    SDL_GLContext m_context;
    SDL_Event m_ev;
};

#endif //BASIC_WINDOW_H_
