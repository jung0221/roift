/**
 * @file orelax_gpu.cuh
 * @brief GPU implementation of ORelax (Oriented Relaxation)
 *
 * This implements the iterative relaxation step of ROIFT on GPU.
 * ORelax is highly parallel - each node can be processed independently.
 */

#ifndef ORELAX_GPU_CUH
#define ORELAX_GPU_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

namespace gft
{
    namespace gpu
    {

        //==============================================================================
        // Configuration
        //==============================================================================

#define ORELAX_BLOCK_SIZE 256
#define ORELAX_MAX_ADJ_SIZE 26 // 3x3x3 - 1 for 3D

        //==============================================================================
        // Data Structures
        //==============================================================================

        /**
         * @brief Pre-computed adjacency for fast GPU access
         */
        struct AdjacencyGPU
        {
            int3 *offsets; // dx, dy, dz for each neighbor
            int size;      // Number of neighbors
        };

        /**
         * @brief ORelax parameters
         */
        struct ORelaxParams
        {
            float Wmax;         // Maximum weight value
            float percentile;   // Orientation percentile (-100 to 100)
            int num_iterations; // Number of relaxation iterations
            int nx, ny, nz;     // Volume dimensions
            int adj_size;       // Adjacency size
        };

        /**
         * @brief ORelax GPU buffers
         */
        struct ORelaxBuffers
        {
            // Scene and mask data
            int *scene_data;    // Original image intensities
            int *mask_nodes;    // Indices of nodes in the mask
            int num_mask_nodes; // Number of nodes in mask

            // Float labels (ping-pong buffers)
            float *flabel_1; // Input float labels
            float *flabel_2; // Output float labels

            // Seed information
            int *seed_nodes;    // Seed node indices
            float *seed_values; // Seed label values (0.0 or 1.0)
            int num_seeds;      // Number of seeds

            // Adjacency (stored as separate arrays for coalesced access)
            int *adj_dx; // Adjacency x offsets
            int *adj_dy; // Adjacency y offsets
            int *adj_dz; // Adjacency z offsets
        };

        //==============================================================================
        // GPU Kernels
        //==============================================================================

        /**
         * @brief Initialize float labels from integer labels
         */
        __global__ void init_flabels_kernel(
            float *flabel,
            const int *labels,
            int num_nodes);

        /**
         * @brief Main ORelax iteration kernel
         *
         * Each thread processes one node in the mask.
         * Computes weighted average of neighbor labels with orientation.
         */
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
            int nx, int ny, int nz);

        /**
         * @brief ORelax kernel with shared memory for adjacency
         */
        __global__ void orelax_iteration_smem_kernel(
            const int *__restrict__ mask_nodes,
            const float *__restrict__ flabel_in,
            float *__restrict__ flabel_out,
            const int *__restrict__ scene_data,
            int adj_size,
            int num_mask_nodes,
            float Wmax,
            float percentile,
            int nx, int ny, int nz);

        /**
         * @brief Reset seed values after each iteration
         */
        __global__ void reset_seeds_kernel(
            float *flabel,
            const int *seed_nodes,
            const float *seed_values,
            int num_seeds);

        /**
         * @brief Convert float labels to binary labels (thresholding)
         */
        __global__ void threshold_labels_kernel(
            const float *flabel,
            int *labels,
            int num_nodes,
            float threshold);

        /**
         * @brief Compute convergence metric (max change between iterations)
         */
        __global__ void compute_max_change_kernel(
            const float *flabel_old,
            const float *flabel_new,
            float *max_change,
            int num_nodes);

        //==============================================================================
        // Device Helper Functions
        //==============================================================================

        /**
         * @brief Compute weight with orientation (inlined for performance)
         */
        __device__ __forceinline__ float compute_orelax_weight(
            int Ip, int Iq, float flabel_q, float Wmax, float percentile)
        {
            // Base weight from intensity difference
            float w = fabsf((float)(Ip - Iq));

            // Orientation based on current label estimate
            float per = (flabel_q < 0.5f) ? percentile : -percentile;

            // Apply orientation
            if (Ip > Iq)
            {
                w *= (1.0f + per / 100.0f);
            }
            else if (Ip < Iq)
            {
                w *= (1.0f - per / 100.0f);
            }

            // Convert to similarity weight
            w = Wmax - w;

            // Apply power function (w^8) for sharpening
            float w2 = w * w;
            float w4 = w2 * w2;
            return w4 * w4;
        }

        /**
         * @brief Check if coordinate is within volume bounds
         */
        __device__ __forceinline__ bool is_valid_coord(
            int x, int y, int z,
            int nx, int ny, int nz)
        {
            return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
        }

        /**
         * @brief Convert 3D coordinate to linear index
         */
        __device__ __forceinline__ int coord_to_index(
            int x, int y, int z,
            int nx, int ny)
        {
            return z * (nx * ny) + y * nx + x;
        }

        /**
         * @brief Convert linear index to 3D coordinates
         */
        __device__ __forceinline__ void index_to_coord(
            int idx,
            int nx, int ny,
            int &x, int &y, int &z)
        {
            int nxy = nx * ny;
            z = idx / nxy;
            int remainder = idx % nxy;
            y = remainder / nx;
            x = remainder % nx;
        }

        //==============================================================================
        // Host Functions
        //==============================================================================

        /**
         * @brief Allocate ORelax GPU buffers
         */
        ORelaxBuffers *allocate_orelax_buffers(
            int num_nodes,
            int num_mask_nodes,
            int num_seeds,
            int adj_size);

        /**
         * @brief Free ORelax GPU buffers
         */
        void free_orelax_buffers(ORelaxBuffers *buffers);

        /**
         * @brief Upload data to GPU buffers
         */
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
            int adj_size);

        /**
         * @brief Download results from GPU
         */
        void download_orelax_results(
            const ORelaxBuffers *buffers,
            float *flabel_out,
            int num_nodes);

        /**
         * @brief Main ORelax GPU function
         *
         * @param buffers Pre-allocated and uploaded GPU buffers
         * @param params ORelax parameters
         * @return Final max change value (for convergence checking)
         */
        float orelax_gpu(
            ORelaxBuffers *buffers,
            const ORelaxParams *params);

        /**
         * @brief Simplified all-in-one ORelax GPU function
         *
         * Handles all allocation, upload, execution, and download.
         *
         * @param scene_data Volume data (host)
         * @param mask_nodes Mask node indices (host)
         * @param num_mask_nodes Number of mask nodes
         * @param flabel_in Initial float labels (host)
         * @param flabel_out Output float labels (host, pre-allocated)
         * @param num_nodes Total number of nodes
         * @param seed_nodes Seed node indices (host)
         * @param seed_values Seed values 0.0 or 1.0 (host)
         * @param num_seeds Number of seeds
         * @param adj_dx Adjacency x offsets (host)
         * @param adj_dy Adjacency y offsets (host)
         * @param adj_dz Adjacency z offsets (host)
         * @param adj_size Adjacency size
         * @param params ORelax parameters
         */
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
            const ORelaxParams *params);

        /**
         * @brief Check if GPU with sufficient capability is available
         */
        bool check_gpu_available();

        /**
         * @brief Get GPU device properties
         */
        void print_gpu_info();

    } // namespace gpu
} // namespace gft

#endif // ORELAX_GPU_CUH
