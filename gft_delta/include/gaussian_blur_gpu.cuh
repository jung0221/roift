/**
 * @file gaussian_blur_gpu.cuh
 * @brief CUDA kernels for Gaussian Blur
 */

#ifndef GAUSSIAN_BLUR_GPU_CUH
#define GAUSSIAN_BLUR_GPU_CUH

#include <cuda_runtime.h>

namespace gft
{
    namespace gpu
    {

        // Block size for blur kernels
        constexpr int BLUR_BLOCK_SIZE = 256;

        // Maximum kernel radius (determines shared memory size)
        constexpr int MAX_KERNEL_RADIUS = 3;
        constexpr int MAX_KERNEL_SIZE = (2 * MAX_KERNEL_RADIUS + 1) * (2 * MAX_KERNEL_RADIUS + 1) * (2 * MAX_KERNEL_RADIUS + 1);

        /**
         * @brief 3D Gaussian Blur kernel
         */
        __global__ void gaussian_blur_3d_kernel(
            const int *__restrict__ input,
            int *__restrict__ output,
            const float *__restrict__ kernel_weights,
            const int *__restrict__ kernel_offsets_x,
            const int *__restrict__ kernel_offsets_y,
            const int *__restrict__ kernel_offsets_z,
            int kernel_size,
            int nx, int ny, int nz);

    } // namespace gpu
} // namespace gft

#endif // GAUSSIAN_BLUR_GPU_CUH
