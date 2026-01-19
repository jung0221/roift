/**
 * @file delta_stepping.h
 * @brief C++ wrapper for Delta-Stepping GPU functions
 *
 * This header provides C++ interface without CUDA dependencies
 * for use in .cpp files. Actual implementation is in .cu files.
 */

#ifndef DELTA_STEPPING_H
#define DELTA_STEPPING_H

namespace gft
{
    namespace gpu
    {

        /**
         * @brief Simplified interface for Delta-Stepping OIFT (C++ callable)
         *
         * @param scene_data 3D volume data (host)
         * @param nx, ny, nz Volume dimensions
         * @param seed_nodes Array of seed node indices
         * @param seed_labels Array of seed labels (0/1)
         * @param num_seeds Number of seeds
         * @param polaridade Orientation parameter
         * @param labels_out Output labels (host, pre-allocated)
         */
        void oift_gpu_delta_stepping(
            const int *scene_data,
            int nx, int ny, int nz,
            const int *seed_nodes,
            const int *seed_labels,
            int num_seeds,
            float polaridade,
            int *labels_out);

        /**
         * @brief Check if GPU with sufficient capability is available
         */
        bool check_gpu_available();

        /**
         * @brief Get GPU device properties
         */
        void print_gpu_info();

        /**
         * @brief Print GPU memory usage
         */
        void print_gpu_memory_usage();

        /**
         * @brief Warm up GPU (first kernel launch is slow)
         */
        void gpu_warmup();

    } // namespace gpu
} // namespace gft

#endif // DELTA_STEPPING_H
