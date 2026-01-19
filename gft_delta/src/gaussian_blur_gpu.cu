/**
 * @file gaussian_blur_gpu.cu
 * @brief CUDA implementation of 3D Gaussian Blur
 *
 * Highly parallel implementation - each voxel processed independently.
 * Uses constant memory for kernel weights.
 */

#include "gaussian_blur_gpu.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

namespace gft
{
    namespace gpu
    {

        // Constant memory for Gaussian kernel (small kernels fit well)
        __constant__ float c_kernel_weights[MAX_KERNEL_SIZE];
        __constant__ int c_kernel_offsets_x[MAX_KERNEL_SIZE];
        __constant__ int c_kernel_offsets_y[MAX_KERNEL_SIZE];
        __constant__ int c_kernel_offsets_z[MAX_KERNEL_SIZE];
        __constant__ int c_kernel_size;

// Error checking macro
#define CUDA_CHECK_BLUR(call)                                      \
    do                                                             \
    {                                                              \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess)                                    \
        {                                                          \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorString(err));            \
        }                                                          \
    } while (0)

        /**
         * @brief 3D Gaussian Blur kernel using constant memory
         *
         * Each thread processes one voxel.
         */
        __global__ void gaussian_blur_3d_kernel(
            const int *__restrict__ input,
            int *__restrict__ output,
            int nx, int ny, int nz)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int num_voxels = nx * ny * nz;

            if (idx >= num_voxels)
                return;

            // Convert linear index to 3D coordinates
            int nxy = nx * ny;
            int z = idx / nxy;
            int rem = idx % nxy;
            int y = rem / nx;
            int x = rem % nx;

            // Compute weighted sum
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int i = 0; i < c_kernel_size; i++)
            {
                int qx = x + c_kernel_offsets_x[i];
                int qy = y + c_kernel_offsets_y[i];
                int qz = z + c_kernel_offsets_z[i];

                // Bounds check
                if (qx >= 0 && qx < nx && qy >= 0 && qy < ny && qz >= 0 && qz < nz)
                {
                    int q = qz * nxy + qy * nx + qx;
                    float w = c_kernel_weights[i];
                    sum += (float)input[q] * w;
                    weight_sum += w;
                }
            }

            // Normalize by actual weights used (handles boundary)
            if (weight_sum > 0.0f)
            {
                output[idx] = (int)roundf(sum / weight_sum * (float)c_kernel_size / weight_sum);
                // Simpler: just use the weighted sum normalized
                output[idx] = (int)roundf(sum / weight_sum);
            }
            else
            {
                output[idx] = input[idx];
            }
        }

        /**
         * @brief Build Spherical Gaussian kernel (matching CPU implementation)
         *
         * Creates a spherical Gaussian kernel with given parameters.
         * K[i] = s * exp(-f * (r^2 / R^2))
         */
        void build_gaussian_kernel(
            float radius,
            float s,
            float f,
            std::vector<float> &weights,
            std::vector<int> &offsets_x,
            std::vector<int> &offsets_y,
            std::vector<int> &offsets_z)
        {
            int R = (int)radius;
            float R2 = radius * radius;

            weights.clear();
            offsets_x.clear();
            offsets_y.clear();
            offsets_z.clear();

            float total_weight = 0.0f;

            // Build spherical kernel
            for (int dz = -R; dz <= R; dz++)
            {
                for (int dy = -R; dy <= R; dy++)
                {
                    for (int dx = -R; dx <= R; dx++)
                    {
                        float r2 = (float)(dx * dx + dy * dy + dz * dz);
                        if (r2 <= R2)
                        {
                            float w = s * expf(-f * (r2 / R2));
                            weights.push_back(w);
                            offsets_x.push_back(dx);
                            offsets_y.push_back(dy);
                            offsets_z.push_back(dz);
                            total_weight += w;
                        }
                    }
                }
            }

            // Normalize weights
            if (total_weight > 0.0f)
            {
                for (size_t i = 0; i < weights.size(); i++)
                {
                    weights[i] /= total_weight;
                }
            }
        }

        void gaussian_blur_gpu(
            const int *input_data,
            int *output_data,
            int nx, int ny, int nz,
            float radius,
            float sigma)
        {
            int num_voxels = nx * ny * nz;

            // Build Gaussian kernel (matching CPU: R=1.0, s=10.0, f=1.0)
            std::vector<float> weights;
            std::vector<int> offsets_x, offsets_y, offsets_z;
            build_gaussian_kernel(radius, sigma, 1.0f, weights, offsets_x, offsets_y, offsets_z);

            int kernel_size = (int)weights.size();

            if (kernel_size > MAX_KERNEL_SIZE)
            {
                fprintf(stderr, "[GaussianBlur-GPU] Kernel size %d exceeds maximum %d\n",
                        kernel_size, MAX_KERNEL_SIZE);
                return;
            }

            // Copy kernel to constant memory
            CUDA_CHECK_BLUR(cudaMemcpyToSymbol(c_kernel_weights, weights.data(), kernel_size * sizeof(float)));
            CUDA_CHECK_BLUR(cudaMemcpyToSymbol(c_kernel_offsets_x, offsets_x.data(), kernel_size * sizeof(int)));
            CUDA_CHECK_BLUR(cudaMemcpyToSymbol(c_kernel_offsets_y, offsets_y.data(), kernel_size * sizeof(int)));
            CUDA_CHECK_BLUR(cudaMemcpyToSymbol(c_kernel_offsets_z, offsets_z.data(), kernel_size * sizeof(int)));
            CUDA_CHECK_BLUR(cudaMemcpyToSymbol(c_kernel_size, &kernel_size, sizeof(int)));

            // Allocate device memory
            int *d_input, *d_output;
            CUDA_CHECK_BLUR(cudaMalloc(&d_input, num_voxels * sizeof(int)));
            CUDA_CHECK_BLUR(cudaMalloc(&d_output, num_voxels * sizeof(int)));

            // Upload input
            CUDA_CHECK_BLUR(cudaMemcpy(d_input, input_data, num_voxels * sizeof(int), cudaMemcpyHostToDevice));

            // Launch kernel
            int grid_size = (num_voxels + BLUR_BLOCK_SIZE - 1) / BLUR_BLOCK_SIZE;
            gaussian_blur_3d_kernel<<<grid_size, BLUR_BLOCK_SIZE>>>(d_input, d_output, nx, ny, nz);

            CUDA_CHECK_BLUR(cudaDeviceSynchronize());

            // Download output
            CUDA_CHECK_BLUR(cudaMemcpy(output_data, d_output, num_voxels * sizeof(int), cudaMemcpyDeviceToHost));

            // Cleanup
            cudaFree(d_input);
            cudaFree(d_output);
        }

        void gaussian_blur_gpu_2x(
            const int *input_data,
            int *output_data,
            int nx, int ny, int nz)
        {
            int num_voxels = nx * ny * nz;

            // Build Gaussian kernel (matching CPU: R=1.0, s=10.0, f=1.0)
            std::vector<float> weights;
            std::vector<int> offsets_x, offsets_y, offsets_z;
            build_gaussian_kernel(1.0f, 10.0f, 1.0f, weights, offsets_x, offsets_y, offsets_z);

            int kernel_size = (int)weights.size();

            // Copy kernel to constant memory
            CUDA_CHECK_BLUR(cudaMemcpyToSymbol(c_kernel_weights, weights.data(), kernel_size * sizeof(float)));
            CUDA_CHECK_BLUR(cudaMemcpyToSymbol(c_kernel_offsets_x, offsets_x.data(), kernel_size * sizeof(int)));
            CUDA_CHECK_BLUR(cudaMemcpyToSymbol(c_kernel_offsets_y, offsets_y.data(), kernel_size * sizeof(int)));
            CUDA_CHECK_BLUR(cudaMemcpyToSymbol(c_kernel_offsets_z, offsets_z.data(), kernel_size * sizeof(int)));
            CUDA_CHECK_BLUR(cudaMemcpyToSymbol(c_kernel_size, &kernel_size, sizeof(int)));

            // Allocate device memory (3 buffers for ping-pong)
            int *d_input, *d_temp, *d_output;
            CUDA_CHECK_BLUR(cudaMalloc(&d_input, num_voxels * sizeof(int)));
            CUDA_CHECK_BLUR(cudaMalloc(&d_temp, num_voxels * sizeof(int)));
            CUDA_CHECK_BLUR(cudaMalloc(&d_output, num_voxels * sizeof(int)));

            // Upload input
            CUDA_CHECK_BLUR(cudaMemcpy(d_input, input_data, num_voxels * sizeof(int), cudaMemcpyHostToDevice));

            // Launch kernel - First pass
            int grid_size = (num_voxels + BLUR_BLOCK_SIZE - 1) / BLUR_BLOCK_SIZE;
            gaussian_blur_3d_kernel<<<grid_size, BLUR_BLOCK_SIZE>>>(d_input, d_temp, nx, ny, nz);
            CUDA_CHECK_BLUR(cudaDeviceSynchronize());

            // Launch kernel - Second pass
            gaussian_blur_3d_kernel<<<grid_size, BLUR_BLOCK_SIZE>>>(d_temp, d_output, nx, ny, nz);
            CUDA_CHECK_BLUR(cudaDeviceSynchronize());

            // Download output
            CUDA_CHECK_BLUR(cudaMemcpy(output_data, d_output, num_voxels * sizeof(int), cudaMemcpyDeviceToHost));

            // Cleanup
            cudaFree(d_input);
            cudaFree(d_temp);
            cudaFree(d_output);
        }

    } // namespace gpu
} // namespace gft
