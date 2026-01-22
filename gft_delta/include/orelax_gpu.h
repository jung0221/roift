/**
 * @file orelax_gpu.h
 * @brief C++ wrapper for ORelax GPU functions
 *
 * This header provides C++ interface without CUDA dependencies
 * for use in .cpp files. Actual implementation is in .cu files.
 */

#ifndef ORELAX_GPU_H
#define ORELAX_GPU_H

namespace gft
{
    namespace gpu
    {

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

#endif // ORELAX_GPU_H
