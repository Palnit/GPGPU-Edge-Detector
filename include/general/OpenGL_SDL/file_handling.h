//
// Created by Palnit on 2024. 01. 16.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_F_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_F_H_

#include "vcpkg_installed/x64-windows/include/SDL2/SDL_surface.h"
#include "GL/glew.h"

namespace FileHandling {

SDL_Surface* LoadImage(const char* file);
GLuint LoadShader(GLenum shaderType, const char* filename);

} // FileHandling

#endif //GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_F_H_
