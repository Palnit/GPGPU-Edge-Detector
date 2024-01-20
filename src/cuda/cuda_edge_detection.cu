//
// Created by Palnit on 2023. 11. 12.
//

#include "include/cuda/cuda_edge_detection.cuh"
#include <cstdint>
#include <cstdio>
#include <math_constants.h>

typedef struct RGBA {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
} RGBA;

__global__ void convertToGreyScale(uint8_t* asd, float* dest, int w, int h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }

    RGBA* color = (RGBA*) (asd + (x * 4) + (y * w * 4));
    *(dest + x + (y * w)) = 0.299 * color->r
        + 0.587 * color->g
        + 0.114 * color->b;

}

__global__ void GetGaussian(float* kernel) {
    uint32_t x = threadIdx.x;
    uint32_t y = threadIdx.y;

    float xp = (((x + 1.f) - (1.f + 2.f)) * ((x + 1.f) - (1.f + 2.f)));
    float yp = (((y + 1.f) - (1.f + 2.f)) * ((y + 1.f) - (1.f + 2.f)));
    *(kernel + x + (y * 5)) =
        (1.f / (2.f * CUDART_PI_F * 4.f)) * expf(-((xp + yp) / (2.f * 4.f)));

}

__global__ void GaussianFilter(float* src,
                               float* dest,
                               float* gauss,
                               int w,
                               int h) {
    int col = blockIdx.x * (blockDim.x - 5 + 1) + threadIdx.x;
    int row = blockIdx.y * (blockDim.y - 5 + 1) + threadIdx.y;
    int col_i = col - 2;
    int row_i = row - 2;

    __shared__ float src_shared[32][32];

    if (col_i >= 0 && col_i < w && row_i >= 0 && row_i < h) {
        src_shared[threadIdx.x][threadIdx.y] = *(src + col_i + (row_i * w));
    } else {
        src_shared[threadIdx.x][threadIdx.y] = 0;
    }

    __syncthreads();
    float sum = 0;

    if (threadIdx.x > 2 - 1 && threadIdx.y > 2 - 1 && threadIdx.x < 32 - 2
        && threadIdx.y < 32 - 2 && col_i < w + 2 && row_i < h + 2) {
        for (int i = -2; i < 3; i++) {
            for (int j = -2; j < 3; j++) {
                sum += src_shared[threadIdx.x + i][threadIdx.y + j]
                    * (*(gauss + (i + 2) + ((j + 2) * 5)));
            }
        }
        *(dest + col_i + (row_i * w)) = sum;
    }

}

__global__ void DetectionOperator(float* src,
                                  float* gradient,
                                  float* tangent,
                                  int w,
                                  int h) {
    int col = blockIdx.x * (blockDim.x - 3 + 1) + threadIdx.x;
    int row = blockIdx.y * (blockDim.y - 3 + 1) + threadIdx.y;
    int col_i = col - 1;
    int row_i = row - 1;

    __shared__ float src_shared[32][32];

    if (col_i >= 0 && col_i < w && row_i >= 0 && row_i < h) {
        src_shared[threadIdx.x][threadIdx.y] = *(src + col_i + (row_i * w));
    } else {
        src_shared[threadIdx.x][threadIdx.y] = 0;
    }

    __syncthreads();

    float SobelX[] = {-1, 0, +1, -2, 0, +2, -1, 0, +1};
    float SobelY[] = {+1, +2, +1, 0, 0, 0, -1, -2, -1};

    float SumX = 0;
    float SumY = 0;

    if (threadIdx.x > 1 - 1 && threadIdx.y > 1 - 1 && threadIdx.x < 32 - 1
        && threadIdx.y < 32 - 1 && col_i < w + 1 && row_i < h + 1) {
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                SumX += src_shared[threadIdx.x + i][threadIdx.y + j]
                    * (*(SobelX + (i + 1) + ((j + 1) * 3)));
                SumY += src_shared[threadIdx.x + i][threadIdx.y + j]
                    * (*(SobelY + (i + 1) + ((j + 1) * 3)));
            }
        }
        *(gradient + col_i + (row_i * w)) = hypotf(SumX, SumY);
        float angle = (atan2(SumX, SumY) * 180.f) / CUDART_PI_F;
        if (angle < 0) {
            angle += 180;
        }
        *(tangent + col_i + (row_i * w)) = angle;
    }

}
__global__ void NonMaximumSuppression(float* gradient_in,
                                      float* gradient_out,
                                      float* tangent,
                                      int w,
                                      int h) {

    int col = blockIdx.x * (blockDim.x - 3 + 1) + threadIdx.x;
    int row = blockIdx.y * (blockDim.y - 3 + 1) + threadIdx.y;
    int col_i = col - 1;
    int row_i = row - 1;

    __shared__ float src_shared[32][32];

    if (col_i >= 0 && col_i < w && row_i >= 0 && row_i < h) {
        src_shared[threadIdx.x][threadIdx.y] =
            *(gradient_in + col_i + (row_i * w));
    } else {
        src_shared[threadIdx.x][threadIdx.y] = 2000;
    }

    __syncthreads();

    if (threadIdx.x <= 1 - 1 || threadIdx.y <= 1 - 1 || threadIdx.x >= 32 - 1
        || threadIdx.y >= 32 - 1 || col_i >= w + 1 || row_i >= h + 1) {
        return;
    }

    float* tangentA = (tangent + col_i + (row_i * w));
    float gradientA = src_shared[threadIdx.x][threadIdx.y];
    float gradientP = 2000;
    float gradientN = 2000;

    if ((0 <= *tangentA && *tangentA < 22.5)
        || (157.5 <= *tangentA && *tangentA <= 180)) {
        gradientP = src_shared[threadIdx.x][threadIdx.y + 1];
        gradientN = src_shared[threadIdx.x][threadIdx.y - 1];
    } else if (22.5 <= *tangentA && *tangentA < 67.5) {
        gradientP = src_shared[threadIdx.x + 1][threadIdx.y - 1];
        gradientN = src_shared[threadIdx.x - 1][threadIdx.y + 1];
    } else if (67.5 <= *tangentA && *tangentA < 112.5) {
        gradientP = src_shared[threadIdx.x + 1][threadIdx.y];
        gradientN = src_shared[threadIdx.x - 1][threadIdx.y];
    } else if (112.5 <= *tangentA && *tangentA < 157.5) {
        gradientP = src_shared[threadIdx.x - 1][threadIdx.y - 1];
        gradientN = src_shared[threadIdx.x + 1][threadIdx.y + 1];
    }

    if (gradientA < gradientN || gradientA < gradientP) {
        gradientA = 0.f;
    }

    *(gradient_out + col_i + (row_i * w)) = gradientA;
}

__global__ void DoubleThreshold(float* gradient_in,
                                float* gradient_out,
                                int w,
                                int h,
                                float high,
                                float low) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }

    if (*(gradient_in + x + (y * w)) >= high) {
        *(gradient_out + x + (y * w)) = 255.f;
    } else if (*(gradient_in + x + (y * w)) < high
        && *(gradient_in + x + (y * w)) >= low) {
        *(gradient_out + x + (y * w)) = 125.f;
    } else {
        *(gradient_out + x + (y * w)) = 0.f;
    }
}

__global__ void Hysteresis(float* gradient_in,
                           float* gradient_out,
                           int w,
                           int h) {
    int col = blockIdx.x * (blockDim.x - 3 + 1) + threadIdx.x;
    int row = blockIdx.y * (blockDim.y - 3 + 1) + threadIdx.y;
    int col_i = col - 1;
    int row_i = row - 1;

    __shared__ float src_shared[32][32];

    if (col_i >= 0 && col_i < w && row_i >= 0 && row_i < h) {
        src_shared[threadIdx.x][threadIdx.y] =
            *(gradient_in + col_i + (row_i * w));
    } else {
        src_shared[threadIdx.x][threadIdx.y] = 0;
    }

    __syncthreads();

    if (threadIdx.x <= 1 - 1 || threadIdx.y <= 1 - 1 || threadIdx.x >= 32 - 1
        || threadIdx.y >= 32 - 1 || col_i >= w + 1 || row_i >= h + 1) {
        return;
    }

    bool strong = false;

    float gradientA = src_shared[threadIdx.x][threadIdx.y];

    *(gradient_out + col_i + (row_i * w)) = gradientA;
    if (gradientA != 125.f) {
        return;
    }
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            if (src_shared[threadIdx.x + j][threadIdx.y + j] == 255.f) {
                strong = true;
            }
        }
    }
    if (strong) {
        gradientA = 255.f;
    } else {
        gradientA = 0.f;
    }

    *(gradient_out + col_i + (row_i * w)) = gradientA;
}

__global__ void CopyBack(uint8_t* src, float* dest, int w, int h) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }
    RGBA* color = (RGBA*) (src + (x * 4) + (y * w * 4));
    color->r = color->g = color->b = *(dest + x + (y * w));

}

void CannyEdgeDetection(uint8_t* src, int w, int h) {
    float* dest1;
    float* dest2;

    float* kernel;
    float* tangent;

    dim3 threads(32, 32);
    dim3 block
        (w / threads.x + (w % threads.x == 0 ? 0 : 1),
         h / threads.y
             + (h % threads.y == 0 ? 0 : 1));

    dim3 block2
        ((w / (threads.x - 5 + 1)) + (w % (threads.x - 5 + 1) == 0 ? 0 : 1),
         (h / (threads.y - 5 + 1))
             + (h % (threads.y - 5 + 1) == 0 ? 0 : 1));
    dim3 block3
        ((w / (threads.x - 3 + 1)) + (w % (threads.x - 3 + 1) == 0 ? 0 : 1),
         (h / (threads.y - 3 + 1))
             + (h % (threads.y - 3 + 1) == 0 ? 0 : 1));

    cudaMalloc((void**) &kernel, sizeof(float) * 25);
    cudaMalloc((void**) &dest1, sizeof(float) * w * h);
    cudaMalloc((void**) &dest2, sizeof(float) * w * h);
    cudaMalloc((void**) &tangent, sizeof(float) * w * h);
    dim3 gauss(5, 5);
    convertToGreyScale<<<block, threads>>>(src, dest1, w, h);
    GetGaussian<<<1, gauss>>>(kernel);
    GaussianFilter<<<block2, threads>>>(dest1, dest2, kernel, w, h);
    DetectionOperator<<<block3, threads>>>(dest2, dest1, tangent, w, h);
    NonMaximumSuppression<<<block3, threads>>>(dest1, dest2, tangent, w, h);
    DoubleThreshold<<<block, threads>>>(dest2, dest1, w, h, 150, 90);
    Hysteresis<<<block3, threads>>>(dest1, dest2, w, h);
    CopyBack<<<block, threads>>>(src, dest2, w, h);
    cudaFree(dest1);
    cudaFree(dest2);
    cudaFree(kernel);
    cudaFree(tangent);
    cudaDeviceSynchronize();
}
