//
// Created by Palnit on 2023. 11. 12.
//

#include "include/cuda/cuda_edge_detection.cuh"
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

__global__ void GetGaussian(float* kernel, int kernelSize, float sigma) {
    uint32_t x = threadIdx.x;
    uint32_t y = threadIdx.y;

    int k = (kernelSize - 1) / 2;

    float xp = (((x + 1.f) - (1.f + k)) * ((x + 1.f) - (1.f + k)));
    float yp = (((y + 1.f) - (1.f + k)) * ((y + 1.f) - (1.f + k)));
    *(kernel + x + (y * kernelSize)) =
        (1.f / (2.f * CUDART_PI_F * sigma * sigma))
            * expf(-((xp + yp) / (2.f * sigma * sigma)));
}

__global__ void GaussianFilter(float* src,
                               float* dest,
                               float* gauss,
                               int kernelSize,
                               int w,
                               int h) {
    int col = blockIdx.x * (blockDim.x - kernelSize + 1) + threadIdx.x;
    int row = blockIdx.y * (blockDim.y - kernelSize + 1) + threadIdx.y;
    int k = (kernelSize - 1) / 2;
    int col_i = col - k;
    int row_i = row - k;

    __shared__ float src_shared[32][32];

    if (col_i >= 0 && col_i < w && row_i >= 0 && row_i < h) {
        src_shared[threadIdx.x][threadIdx.y] = *(src + col_i + (row_i * w));
    } else {
        src_shared[threadIdx.x][threadIdx.y] = 0;
    }

    __syncthreads();
    float sum = 0;

    if (threadIdx.x > k - 1 && threadIdx.y > k - 1 && threadIdx.x < 32 - k
        && threadIdx.y < 32 - k && col_i < w + k && row_i < h + k) {

        for (int i = -k; i <= k; i++) {
            for (int j = -k; j <= k; j++) {
                sum += src_shared[threadIdx.x + i][threadIdx.y + j]
                    * (*(gauss + (i + k) + ((j + k) * kernelSize)));
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
    if (x == 0 && y == 0) {
    }

}

void CudaDetector::CannyEdgeDetection() {
    float* dest1;
    float* dest2;

    float* kernel;
    float* tangent;

    dim3 threads(32, 32);
    dim3 block
        (m_w / threads.x + (m_w % threads.x == 0 ? 0 : 1),
         m_h / threads.y
             + (m_h % threads.y == 0 ? 0 : 1));

    dim3 block2
        ((m_w / (threads.x - m_gaussKernelSize + 1))
             + (m_w % (threads.x - m_gaussKernelSize + 1) == 0 ? 0 : 1),
         (m_h / (threads.y - m_gaussKernelSize + 1))
             + (m_h % (threads.y - m_gaussKernelSize + 1) == 0 ? 0 : 1));
    dim3 block3
        ((m_w / (threads.x - 3 + 1)) + (m_w % (threads.x - 3 + 1) == 0 ? 0 : 1),
         (m_h / (threads.y - 3 + 1))
             + (m_h % (threads.y - 3 + 1) == 0 ? 0 : 1));

    cudaMalloc((void**) &kernel, sizeof(float) * 25);
    cudaMalloc((void**) &dest1, sizeof(float) * m_w * m_h);
    cudaMalloc((void**) &dest2, sizeof(float) * m_w * m_h);
    cudaMalloc((void**) &tangent, sizeof(float) * m_w * m_h);
    dim3 gauss(m_gaussKernelSize, m_gaussKernelSize);
    convertToGreyScale<<<block, threads>>>(m_src, dest1, m_w, m_h);
    GetGaussian<<<1, gauss>>>(kernel, m_gaussKernelSize, m_standardDeviation);
    GaussianFilter<<<block2, threads>>>(dest1,
                                        dest2,
                                        kernel,
                                        m_gaussKernelSize,
                                        m_w,
                                        m_h);
    DetectionOperator<<<block3, threads>>>(dest2, dest1, tangent, m_w, m_h);
    NonMaximumSuppression<<<block3, threads>>>(dest1, dest2, tangent, m_w, m_h);
    DoubleThreshold<<<block, threads>>>(dest2, dest1, m_w, m_h, m_high, m_low);
    Hysteresis<<<block3, threads>>>(dest1, dest2, m_w, m_h);
    CopyBack<<<block, threads>>>(m_src, dest2, m_w, m_h);
    cudaFree(dest1);
    cudaFree(dest2);
    cudaFree(kernel);
    cudaFree(tangent);
    cudaDeviceSynchronize();
}
