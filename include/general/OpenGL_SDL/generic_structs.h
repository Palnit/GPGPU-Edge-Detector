//
// Created by Palnit on 2024. 01. 17.
//

#ifndef GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_GENERIC_STRUCTS_H_
#define GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_GENERIC_STRUCTS_H_

#include <cstdint>

typedef struct RGBA {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
} RGBA;

typedef union Color {
    uint32_t raw;
    RGBA channels;
} Color;

class Time {
public:
    Time() = delete;
    ~Time() = delete;
    static inline uint64_t ElapsedTime = 0;
    static inline uint64_t DeltaTime = 0;
    static inline double FPS = 0;
    static inline double Ms = 0;
};

namespace ErrorHandling {
void HandelSDLError(const char* type);

}
#endif //GPGPU_EDGE_DETECTOR_INCLUDE_GENERAL_GENERIC_STRUCTS_H_
