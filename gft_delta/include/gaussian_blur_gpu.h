/**
 * @file gaussian_blur_gpu.h
 * @brief C++ wrapper for GPU Gaussian Blur
 *
 * Provides C++ interface without CUDA dependencies.
 */

#ifndef GAUSSIAN_BLUR_GPU_H
#define GAUSSIAN_BLUR_GPU_H

namespace gft
{
    namespace gpu
    {

        /**
         * @brief GPU Gaussian Blur for 3D volumes
         *
         * @param input_data Input volume data (host)
         * @param output_data Output volume data (host, pre-allocated)
         * @param nx, ny, nz Volume dimensions
         * @param radius Kernel radius (default 1.0)
         * @param sigma Gaussian sigma parameter (default 10.0)
         */
        void gaussian_blur_gpu(
            const int *input_data,
            int *output_data,
            int nx, int ny, int nz,
            float radius = 1.0f,
            float sigma = 10.0f);

        /**
         * @brief GPU Gaussian Blur applied twice (matching CPU pipeline)
         *
         * @param input_data Input volume data (host)
         * @param output_data Output volume data (host, pre-allocated)
         * @param nx, ny, nz Volume dimensions
         */
        void gaussian_blur_gpu_2x(
            const int *input_data,
            int *output_data,
            int nx, int ny, int nz);

    } // namespace gpu
} // namespace gft

#endif // GAUSSIAN_BLUR_GPU_H
