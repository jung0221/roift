/**
 * @file orelax_gpu.cu
 * @brief CUDA implementation of ORelax (Oriented Relaxation)
 *
 * Highly parallel implementation - each node processed independently.
 * Expected speedup: 30-60x vs CPU single-thread
 */

#include "orelax_gpu.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <cstdio>
#include <cfloat>

namespace gft
{
    namespace gpu
    {

        //==============================================================================
        // Constant Memory for Adjacency
        //==============================================================================

        __constant__ int orelax_adj_dx[ORELAX_MAX_ADJ_SIZE];
        __constant__ int orelax_adj_dy[ORELAX_MAX_ADJ_SIZE];
        __constant__ int orelax_adj_dz[ORELAX_MAX_ADJ_SIZE];
        __constant__ int orelax_adj_size;

        //==============================================================================
        // Kernel Implementations
        //==============================================================================

        __global__ void init_flabels_kernel(
            float *flabel,
            const int *labels,
            int num_nodes)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_nodes)
                return;

            flabel[idx] = (labels[idx] == 1) ? 1.0f : 0.0f;
        }

        __global__ void orelax_iteration_kernel(
            const int *__restrict__ mask_nodes,
            const float *__restrict__ flabel_in,
            float *__restrict__ flabel_out,
            const int *__restrict__ scene_data,
            const int *__restrict__ adj_dx,
            const int *__restrict__ adj_dy,
            const int *__restrict__ adj_dz,
            int adj_size,
            int num_mask_nodes,
            float Wmax,
            float percentile,
            int nx, int ny, int nz)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= num_mask_nodes)
                return;

            int p = mask_nodes[tid];

            // Convert linear index to 3D coordinates
            int nxy = nx * ny;
            int pz = p / nxy;
            int rem = p % nxy;
            int py = rem / nx;
            int px = rem % nx;

            int Ip = scene_data[p];

            float sum_w = 0.0f;
            float sum_weighted = 0.0f;

// Process all neighbors
#pragma unroll
            for (int i = 0; i < adj_size; i++)
            {
                int qx = px + adj_dx[i];
                int qy = py + adj_dy[i];
                int qz = pz + adj_dz[i];

                // Bounds check
                if (qx >= 0 && qx < nx && qy >= 0 && qy < ny && qz >= 0 && qz < nz)
                {
                    int q = qz * nxy + qy * nx + qx;
                    int Iq = scene_data[q];
                    float flabel_q = flabel_in[q];

                    // Compute oriented weight
                    float w = compute_orelax_weight(Ip, Iq, flabel_q, Wmax, percentile);

                    sum_w += w;
                    sum_weighted += w * flabel_q;
                }
            }

            // Compute new label value
            if (sum_w > 0.0f)
            {
                flabel_out[p] = sum_weighted / sum_w;
            }
            else
            {
                flabel_out[p] = flabel_in[p];
            }
        }

        __global__ void orelax_iteration_smem_kernel(
            const int *__restrict__ mask_nodes,
            const float *__restrict__ flabel_in,
            float *__restrict__ flabel_out,
            const int *__restrict__ scene_data,
            int adj_size,
            int num_mask_nodes,
            float Wmax,
            float percentile,
            int nx, int ny, int nz)
        {
            // Use constant memory adjacency
            extern __shared__ int s_adj[];
            int *s_adj_dx = s_adj;
            int *s_adj_dy = s_adj + ORELAX_MAX_ADJ_SIZE;
            int *s_adj_dz = s_adj + 2 * ORELAX_MAX_ADJ_SIZE;

            // Load adjacency to shared memory (first warp)
            if (threadIdx.x < adj_size)
            {
                s_adj_dx[threadIdx.x] = orelax_adj_dx[threadIdx.x];
                s_adj_dy[threadIdx.x] = orelax_adj_dy[threadIdx.x];
                s_adj_dz[threadIdx.x] = orelax_adj_dz[threadIdx.x];
            }
            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= num_mask_nodes)
                return;

            int p = mask_nodes[tid];

            // Convert linear index to 3D coordinates
            int nxy = nx * ny;
            int pz = p / nxy;
            int rem = p % nxy;
            int py = rem / nx;
            int px = rem % nx;

            int Ip = scene_data[p];

            float sum_w = 0.0f;
            float sum_weighted = 0.0f;

            // Process all neighbors using shared memory adjacency
            for (int i = 0; i < adj_size; i++)
            {
                int qx = px + s_adj_dx[i];
                int qy = py + s_adj_dy[i];
                int qz = pz + s_adj_dz[i];

                // Bounds check
                if (qx >= 0 && qx < nx && qy >= 0 && qy < ny && qz >= 0 && qz < nz)
                {
                    int q = qz * nxy + qy * nx + qx;
                    int Iq = scene_data[q];
                    float flabel_q = flabel_in[q];

                    // Compute oriented weight
                    float w = compute_orelax_weight(Ip, Iq, flabel_q, Wmax, percentile);

                    sum_w += w;
                    sum_weighted += w * flabel_q;
                }
            }

            // Compute new label value
            if (sum_w > 0.0f)
            {
                flabel_out[p] = sum_weighted / sum_w;
            }
            else
            {
                flabel_out[p] = flabel_in[p];
            }
        }

        __global__ void reset_seeds_kernel(
            float *flabel,
            const int *seed_nodes,
            const float *seed_values,
            int num_seeds)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_seeds)
                return;

            int node = seed_nodes[idx];
            flabel[node] = seed_values[idx];
        }

        __global__ void threshold_labels_kernel(
            const float *flabel,
            int *labels,
            int num_nodes,
            float threshold)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_nodes)
                return;

            labels[idx] = (flabel[idx] >= threshold) ? 1 : 0;
        }

        // Functor for max absolute difference
        struct MaxDiffFunctor
        {
            const float *old_labels;
            const float *new_labels;

            __device__ float operator()(int idx) const
            {
                return fabsf(new_labels[idx] - old_labels[idx]);
            }
        };

        __global__ void compute_max_change_kernel(
            const float *flabel_old,
            const float *flabel_new,
            float *max_change,
            int num_nodes)
        {
            __shared__ float s_max[ORELAX_BLOCK_SIZE];

            int tid = threadIdx.x;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            // Compute local max
            float local_max = 0.0f;
            while (idx < num_nodes)
            {
                float diff = fabsf(flabel_new[idx] - flabel_old[idx]);
                local_max = fmaxf(local_max, diff);
                idx += blockDim.x * gridDim.x;
            }

            s_max[tid] = local_max;
            __syncthreads();

            // Reduction in shared memory
            for (int s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (tid < s)
                {
                    s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
                }
                __syncthreads();
            }

            // Write block result
            if (tid == 0)
            {
                atomicMax(reinterpret_cast<int *>(max_change), __float_as_int(s_max[0]));
            }
        }

        //==============================================================================
        // Host Implementation
        //==============================================================================

        ORelaxBuffers *allocate_orelax_buffers(
            int num_nodes,
            int num_mask_nodes,
            int num_seeds,
            int adj_size)
        {
            ORelaxBuffers *buffers = new ORelaxBuffers();
            buffers->num_mask_nodes = num_mask_nodes;
            buffers->num_seeds = num_seeds;

            // Allocate device memory
            cudaMalloc(&buffers->scene_data, num_nodes * sizeof(int));
            cudaMalloc(&buffers->mask_nodes, num_mask_nodes * sizeof(int));
            cudaMalloc(&buffers->flabel_1, num_nodes * sizeof(float));
            cudaMalloc(&buffers->flabel_2, num_nodes * sizeof(float));
            cudaMalloc(&buffers->seed_nodes, num_seeds * sizeof(int));
            cudaMalloc(&buffers->seed_values, num_seeds * sizeof(float));
            cudaMalloc(&buffers->adj_dx, adj_size * sizeof(int));
            cudaMalloc(&buffers->adj_dy, adj_size * sizeof(int));
            cudaMalloc(&buffers->adj_dz, adj_size * sizeof(int));

            return buffers;
        }

        void free_orelax_buffers(ORelaxBuffers *buffers)
        {
            if (buffers == nullptr)
                return;

            cudaFree(buffers->scene_data);
            cudaFree(buffers->mask_nodes);
            cudaFree(buffers->flabel_1);
            cudaFree(buffers->flabel_2);
            cudaFree(buffers->seed_nodes);
            cudaFree(buffers->seed_values);
            cudaFree(buffers->adj_dx);
            cudaFree(buffers->adj_dy);
            cudaFree(buffers->adj_dz);

            delete buffers;
        }

        void upload_orelax_data(
            ORelaxBuffers *buffers,
            const int *scene_data,
            const int *mask_nodes,
            int num_mask_nodes,
            const float *initial_labels,
            int num_nodes,
            const int *seed_nodes,
            const float *seed_values,
            int num_seeds,
            const int *adj_dx,
            const int *adj_dy,
            const int *adj_dz,
            int adj_size)
        {
            cudaMemcpy(buffers->scene_data, scene_data,
                       num_nodes * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(buffers->mask_nodes, mask_nodes,
                       num_mask_nodes * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(buffers->flabel_1, initial_labels,
                       num_nodes * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(buffers->seed_nodes, seed_nodes,
                       num_seeds * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(buffers->seed_values, seed_values,
                       num_seeds * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(buffers->adj_dx, adj_dx,
                       adj_size * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(buffers->adj_dy, adj_dy,
                       adj_size * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(buffers->adj_dz, adj_dz,
                       adj_size * sizeof(int), cudaMemcpyHostToDevice);

            // Also copy to constant memory
            cudaMemcpyToSymbol(orelax_adj_dx, adj_dx, adj_size * sizeof(int));
            cudaMemcpyToSymbol(orelax_adj_dy, adj_dy, adj_size * sizeof(int));
            cudaMemcpyToSymbol(orelax_adj_dz, adj_dz, adj_size * sizeof(int));
            cudaMemcpyToSymbol(orelax_adj_size, &adj_size, sizeof(int));

            buffers->num_mask_nodes = num_mask_nodes;
            buffers->num_seeds = num_seeds;
        }

        void download_orelax_results(
            const ORelaxBuffers *buffers,
            float *flabel_out,
            int num_nodes)
        {
            cudaMemcpy(flabel_out, buffers->flabel_1,
                       num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
        }

        float orelax_gpu(
            ORelaxBuffers *buffers,
            const ORelaxParams *params)
        {
            int num_mask = buffers->num_mask_nodes;
            int num_seeds = buffers->num_seeds;

            int grid_mask = (num_mask + ORELAX_BLOCK_SIZE - 1) / ORELAX_BLOCK_SIZE;
            int grid_seeds = (num_seeds + ORELAX_BLOCK_SIZE - 1) / ORELAX_BLOCK_SIZE;

            // Shared memory size for adjacency
            size_t smem_size = 3 * ORELAX_MAX_ADJ_SIZE * sizeof(int);

            float *flabel_in = buffers->flabel_1;
            float *flabel_out = buffers->flabel_2;

            float max_change = 0.0f;

            for (int iter = 0; iter < params->num_iterations; iter++)
            {
                // Main relaxation kernel
                orelax_iteration_smem_kernel<<<grid_mask, ORELAX_BLOCK_SIZE, smem_size>>>(
                    buffers->mask_nodes,
                    flabel_in,
                    flabel_out,
                    buffers->scene_data,
                    params->adj_size,
                    num_mask,
                    params->Wmax,
                    params->percentile,
                    params->nx, params->ny, params->nz);

                // Reset seeds to their original values
                reset_seeds_kernel<<<grid_seeds, ORELAX_BLOCK_SIZE>>>(
                    flabel_out,
                    buffers->seed_nodes,
                    buffers->seed_values,
                    num_seeds);

                // Swap buffers
                float *temp = flabel_in;
                flabel_in = flabel_out;
                flabel_out = temp;
            }

            // Make sure flabel_1 has the final result
            if (flabel_in != buffers->flabel_1)
            {
                cudaMemcpy(buffers->flabel_1, flabel_in,
                           params->nx * params->ny * params->nz * sizeof(float),
                           cudaMemcpyDeviceToDevice);
            }

            cudaDeviceSynchronize();

            return max_change;
        }

        void orelax_gpu_simple(
            const int *scene_data,
            const int *mask_nodes,
            int num_mask_nodes,
            const float *flabel_in,
            float *flabel_out,
            int num_nodes,
            const int *seed_nodes,
            const float *seed_values,
            int num_seeds,
            const int *adj_dx,
            const int *adj_dy,
            const int *adj_dz,
            int adj_size,
            const ORelaxParams *params)
        {
            // Allocate buffers
            ORelaxBuffers *buffers = allocate_orelax_buffers(
                num_nodes, num_mask_nodes, num_seeds, adj_size);

            // Upload data
            upload_orelax_data(
                buffers,
                scene_data, mask_nodes, num_mask_nodes,
                flabel_in, num_nodes,
                seed_nodes, seed_values, num_seeds,
                adj_dx, adj_dy, adj_dz, adj_size);

            // Run ORelax
            orelax_gpu(buffers, params);

            // Download results
            download_orelax_results(buffers, flabel_out, num_nodes);

            // Cleanup
            free_orelax_buffers(buffers);
        }

        bool check_gpu_available()
        {
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);

            if (err != cudaSuccess || device_count == 0)
            {
                return false;
            }

            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);

            // Require compute capability 3.5 or higher
            return (prop.major > 3 || (prop.major == 3 && prop.minor >= 5));
        }

        void print_gpu_info()
        {
            int device_count = 0;
            cudaGetDeviceCount(&device_count);

            printf("CUDA Devices: %d\n", device_count);

            for (int i = 0; i < device_count; i++)
            {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);

                printf("\nDevice %d: %s\n", i, prop.name);
                printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
                printf("  Total memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
                printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
                printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
                printf("  Warp size: %d\n", prop.warpSize);
            }
        }

    } // namespace gpu
} // namespace gft
