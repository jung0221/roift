/**
 * @file delta_stepping.cu
 * @brief CUDA implementation of Delta-Stepping OIFT
 *
 * Implements the Delta-Stepping algorithm for parallel shortest path
 * computation adapted for Oriented Image Foresting Transform.
 */

#include "delta_stepping.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <queue>

namespace gft
{
    namespace gpu
    {

        //==============================================================================
        // Constant Memory for Adjacency
        //==============================================================================

        // Adjacency stored in constant memory for fast access
        __constant__ int c_adj_dx[27];
        __constant__ int c_adj_dy[27];
        __constant__ int c_adj_dz[27];
        __constant__ int c_adj_size;

        //==============================================================================
        // Kernel Implementations
        //==============================================================================

        // Forward declaration for Bellman-Ford kernel
        __global__ void oift_bellman_ford_kernel(
            const int *scene_data,
            unsigned long long *cost_label,
            int *updated,
            float polaridade,
            int nx, int ny, int nz,
            int num_nodes);

        __global__ void init_seeds_kernel(
            float *costs,
            int *labels,
            int *roots,
            const int *seed_nodes,
            const int *seed_labels,
            int num_seeds,
            int num_nodes)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            // First, initialize all nodes with infinite cost
            if (idx < num_nodes)
            {
                costs[idx] = FLT_MAX;
                labels[idx] = -1; // Undefined
                roots[idx] = -1;
            }

            // Then, set seed costs to 0
            if (idx < num_seeds)
            {
                int node = seed_nodes[idx];
                costs[node] = 0.0f;
                labels[node] = seed_labels[idx];
                roots[node] = node;
            }
        }

        __global__ void assign_buckets_kernel(
            const float *costs,
            int *bucket_indices,
            const int *active_mask,
            float delta,
            int num_nodes)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_nodes)
                return;

            if (active_mask == nullptr || active_mask[idx])
            {
                float cost = costs[idx];
                if (cost < FLT_MAX)
                {
                    bucket_indices[idx] = (int)(cost / delta);
                }
                else
                {
                    bucket_indices[idx] = INT_MAX;
                }
            }
            else
            {
                bucket_indices[idx] = INT_MAX;
            }
        }

        __global__ void oift_relax_kernel(
            const int *active_nodes,
            int num_active,
            const int *row_offsets,
            const int *col_indices,
            const int *scene_data,
            float *costs,
            int *labels,
            int *roots,
            int *predecessors,
            int *updated_flags,
            float Wmax,
            float polaridade,
            int nx, int ny, int nz)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= num_active)
                return;

            int p = active_nodes[tid];
            int label_p = labels[p];
            int root_p = roots[p];

            // Skip if no valid label
            if (label_p < 0)
                return;

            // Get node coordinates
            int nxy = nx * ny;
            int pz = p / nxy;
            int rem = p % nxy;
            int py = rem / nx;
            int px = rem % nx;

            int Ip = scene_data[p];

            // Process all neighbors
            for (int i = 0; i < c_adj_size; i++)
            {
                int qx = px + c_adj_dx[i];
                int qy = py + c_adj_dy[i];
                int qz = pz + c_adj_dz[i];

                // Bounds check
                if (qx < 0 || qx >= nx || qy < 0 || qy >= ny || qz < 0 || qz >= nz)
                {
                    continue;
                }

                int q = qz * nxy + qy * nx + qx;
                int Iq = scene_data[q];

                // OIFT Edge Weight Formula (matching CPU implementation)
                float w = fabsf((float)(Ip - Iq));
                float per_pq = (label_p > 0) ? polaridade : -polaridade;

                if (Ip > Iq)
                {
                    w = w * (1.0f + per_pq);
                }
                else if (Ip < Iq)
                {
                    w = w * (1.0f - per_pq);
                }

                // OIFT: new cos IS the edge weight (not path-max)
                float new_cost = fmaxf(costs[p], w);

                // Atomic compare-and-swap for float minimum
                float old_cost = costs[q];
                while (new_cost < old_cost)
                {
                    int assumed = __float_as_int(old_cost);
                    int old_int = atomicCAS(reinterpret_cast<int *>(&costs[q]),
                                            assumed, __float_as_int(new_cost));
                    if (old_int == assumed)
                    {
                        labels[q] = label_p;
                        roots[q] = root_p;
                        if (predecessors != nullptr)
                        {
                            predecessors[q] = p;
                        }
                        atomicExch(&updated_flags[0], 1);
                        break;
                    }
                    old_cost = __int_as_float(old_int);
                }
            }
        }

        __global__ void process_bucket_kernel(
            const int *bucket_nodes,
            int bucket_size,
            const int *scene_data,
            float *costs,
            int *labels,
            int *roots,
            int *updated_flags,
            float Wmax,
            float polaridade,
            float delta,
            int nx, int ny, int nz)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= bucket_size)
                return;

            int p = bucket_nodes[tid];
            int label_p = labels[p];
            int root_p = roots[p];

            // Skip if this node doesn't have a valid label
            if (label_p < 0)
                return;

            // Get node coordinates
            int nxy = nx * ny;
            int pz = p / nxy;
            int rem = p % nxy;
            int py = rem / nx;
            int px = rem % nx;

            int Ip = scene_data[p];

            // Current bucket for light/heavy edge classification
            float cost_p = costs[p];
            int current_bucket = (int)(cost_p / delta);

            // Process all neighbors
            for (int i = 0; i < c_adj_size; i++)
            {
                int qx = px + c_adj_dx[i];
                int qy = py + c_adj_dy[i];
                int qz = pz + c_adj_dz[i];

                // Bounds check
                if (qx < 0 || qx >= nx || qy < 0 || qy >= ny || qz < 0 || qz >= nz)
                {
                    continue;
                }

                int q = qz * nxy + qy * nx + qx;
                int Iq = scene_data[q];

                // ============================================================
                // OIFT Edge Weight Formula (from CPU implementation):
                // w = |I(p) - I(q)|
                // if label(p) > 0: per_pq = +polaridade, else per_pq = -polaridade
                // if I(p) > I(q): w *= (1 + per_pq/100)
                // if I(p) < I(q): w *= (1 - per_pq/100)
                // ============================================================
                float w = fabsf((float)(Ip - Iq));

                // polaridade is already in [-1, 1] range from caller, but CPU uses /100
                // Adjust: polaridade here is already fraction, CPU divides by 100
                float per_pq = (label_p > 0) ? polaridade : -polaridade;

                if (Ip > Iq)
                {
                    w = w * (1.0f + per_pq);
                }
                else if (Ip < Iq)
                {
                    w = w * (1.0f - per_pq);
                }
                // If Ip == Iq, w stays as |Ip - Iq| = 0

                // OIFT: The new cost IS the edge weight (not path-max!)
                // We want to minimize edge weight to reach each node
                float new_cost = w;

                // Atomic compare-and-swap loop for float min
                float old_cost = costs[q];
                while (new_cost < old_cost)
                {
                    int assumed = __float_as_int(old_cost);
                    int old_int = atomicCAS(reinterpret_cast<int *>(&costs[q]),
                                            assumed, __float_as_int(new_cost));
                    if (old_int == assumed)
                    {
                        // Successfully updated - also update label and root
                        labels[q] = label_p;
                        roots[q] = root_p;

                        // Check if this is a light edge (within delta of current bucket)
                        int new_bucket = (int)(new_cost / delta);
                        if (new_bucket <= current_bucket)
                        {
                            // Mark for re-processing in current bucket
                            atomicExch(&updated_flags[0], 1);
                        }
                        break;
                    }
                    old_cost = __int_as_float(old_int);
                }
            }
        }

        __global__ void find_min_bucket_kernel(
            const float *costs,
            const int *labels,
            int *min_bucket,
            float delta,
            int num_nodes)
        {
            __shared__ int s_min_bucket;

            if (threadIdx.x == 0)
            {
                s_min_bucket = INT_MAX;
            }
            __syncthreads();

            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            // Find nodes with finite cost (regardless of label, since seeds have labels)
            if (idx < num_nodes && costs[idx] < FLT_MAX)
            {
                int bucket = (int)(costs[idx] / delta);
                atomicMin(&s_min_bucket, bucket);
            }

            __syncthreads();

            if (threadIdx.x == 0)
            {
                atomicMin(min_bucket, s_min_bucket);
            }
        }

        __global__ void extract_bucket_nodes_kernel(
            const float *costs,
            const int *labels,
            int *bucket_nodes,
            int *bucket_count,
            int target_bucket,
            float delta,
            int num_nodes)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_nodes)
                return;

            // Extract nodes with finite cost in target bucket (they should have label assigned)
            if (costs[idx] < FLT_MAX && labels[idx] >= 0)
            {
                int bucket = (int)(costs[idx] / delta);
                if (bucket == target_bucket)
                {
                    int pos = atomicAdd(bucket_count, 1);
                    bucket_nodes[pos] = idx;
                }
            }
        }

        __global__ void count_unsettled_kernel(
            const int *labels,
            int *count,
            int num_nodes)
        {
            __shared__ int s_count;

            if (threadIdx.x == 0)
            {
                s_count = 0;
            }
            __syncthreads();

            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < num_nodes && labels[idx] < 0)
            {
                atomicAdd(&s_count, 1);
            }

            __syncthreads();

            if (threadIdx.x == 0)
            {
                atomicAdd(count, s_count);
            }
        }

        //==============================================================================
        // Host Implementation
        //==============================================================================

        GPUGraph *allocate_gpu_graph(int num_nodes, int num_edges, int adj_size)
        {
            GPUGraph *graph = new GPUGraph();
            graph->num_nodes = num_nodes;
            graph->num_edges = num_edges;
            graph->adj_size = adj_size;

            CUDA_CHECK(cudaMalloc(&graph->costs, num_nodes * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&graph->labels, num_nodes * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&graph->roots, num_nodes * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&graph->predecessors, num_nodes * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&graph->scene_data, num_nodes * sizeof(int)));

            CUDA_CHECK(cudaMalloc(&graph->adj_offsets_x, adj_size * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&graph->adj_offsets_y, adj_size * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&graph->adj_offsets_z, adj_size * sizeof(int)));

            return graph;
        }

        void free_gpu_graph(GPUGraph *graph)
        {
            if (graph == nullptr)
                return;

            cudaFree(graph->costs);
            cudaFree(graph->labels);
            cudaFree(graph->roots);
            cudaFree(graph->predecessors);
            cudaFree(graph->scene_data);
            cudaFree(graph->adj_offsets_x);
            cudaFree(graph->adj_offsets_y);
            cudaFree(graph->adj_offsets_z);

            delete graph;
        }

        void convert_adjacency_to_gpu(
            GPUGraph *gpu_graph,
            const int *adj_x, const int *adj_y, const int *adj_z,
            int adj_size)
        {
            // Copy to constant memory for fast access
            CUDA_CHECK(cudaMemcpyToSymbol(c_adj_dx, adj_x, adj_size * sizeof(int)));
            CUDA_CHECK(cudaMemcpyToSymbol(c_adj_dy, adj_y, adj_size * sizeof(int)));
            CUDA_CHECK(cudaMemcpyToSymbol(c_adj_dz, adj_z, adj_size * sizeof(int)));
            CUDA_CHECK(cudaMemcpyToSymbol(c_adj_size, &adj_size, sizeof(int)));

            // Also copy to graph structure
            CUDA_CHECK(cudaMemcpy(gpu_graph->adj_offsets_x, adj_x,
                                  adj_size * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(gpu_graph->adj_offsets_y, adj_y,
                                  adj_size * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(gpu_graph->adj_offsets_z, adj_z,
                                  adj_size * sizeof(int), cudaMemcpyHostToDevice));
        }

        int delta_stepping_oift(
            GPUGraph *gpu_graph,
            const GPUSeeds *seeds,
            const DeltaConfig *config,
            int *labels_out)
        {
            int num_nodes = gpu_graph->num_nodes;
            float delta = config->delta;
            float Wmax = config->Wmax;
            float polaridade = config->polaridade;

            // Device memory for seeds
            int *d_seed_nodes;
            int *d_seed_labels;
            CUDA_CHECK(cudaMalloc(&d_seed_nodes, seeds->num_seeds * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_seed_labels, seeds->num_seeds * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_seed_nodes, seeds->seed_nodes,
                                  seeds->num_seeds * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_seed_labels, seeds->seed_labels,
                                  seeds->num_seeds * sizeof(int), cudaMemcpyHostToDevice));

            // Work arrays
            int *d_bucket_nodes;
            int *d_bucket_count;
            int *d_min_bucket;
            int *d_updated_flag;

            CUDA_CHECK(cudaMalloc(&d_bucket_nodes, num_nodes * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_bucket_count, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_min_bucket, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_updated_flag, sizeof(int)));

            // Initialize
            int grid_nodes = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int grid_seeds = (seeds->num_seeds + BLOCK_SIZE - 1) / BLOCK_SIZE;

            init_seeds_kernel<<<grid_nodes, BLOCK_SIZE>>>(
                gpu_graph->costs, gpu_graph->labels, gpu_graph->roots,
                d_seed_nodes, d_seed_labels, seeds->num_seeds, num_nodes);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Main Delta-Stepping loop
            int iterations = 0;
            int max_iterations = config->max_iterations > 0 ? config->max_iterations : MAX_DELTA_ITERATIONS;

            while (iterations < max_iterations)
            {
                // Find minimum non-empty bucket
                int h_min_bucket = INT_MAX;
                CUDA_CHECK(cudaMemcpy(d_min_bucket, &h_min_bucket, sizeof(int), cudaMemcpyHostToDevice));

                find_min_bucket_kernel<<<grid_nodes, BLOCK_SIZE>>>(
                    gpu_graph->costs, gpu_graph->labels, d_min_bucket, delta, num_nodes);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(&h_min_bucket, d_min_bucket, sizeof(int), cudaMemcpyDeviceToHost));

                if (h_min_bucket == INT_MAX)
                {
                    // All nodes processed
                    break;
                }

                // Process current bucket (with light edge iterations)
                int light_iterations = 0;
                int max_light = 100; // Limit light edge iterations

                while (light_iterations < max_light)
                {
                    // Extract nodes in current bucket
                    int h_bucket_count = 0;
                    CUDA_CHECK(cudaMemcpy(d_bucket_count, &h_bucket_count, sizeof(int), cudaMemcpyHostToDevice));

                    extract_bucket_nodes_kernel<<<grid_nodes, BLOCK_SIZE>>>(
                        gpu_graph->costs, gpu_graph->labels, d_bucket_nodes, d_bucket_count,
                        h_min_bucket, delta, num_nodes);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaMemcpy(&h_bucket_count, d_bucket_count, sizeof(int), cudaMemcpyDeviceToHost));

                    if (h_bucket_count == 0)
                    {
                        break;
                    }

                    // Reset update flag
                    int h_updated = 0;
                    CUDA_CHECK(cudaMemcpy(d_updated_flag, &h_updated, sizeof(int), cudaMemcpyHostToDevice));

                    // Process bucket
                    int grid_bucket = (h_bucket_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    process_bucket_kernel<<<grid_bucket, BLOCK_SIZE>>>(
                        d_bucket_nodes, h_bucket_count,
                        gpu_graph->scene_data, gpu_graph->costs,
                        gpu_graph->labels, gpu_graph->roots,
                        d_updated_flag, Wmax, polaridade, delta,
                        gpu_graph->nx, gpu_graph->ny, gpu_graph->nz);
                    CUDA_CHECK(cudaDeviceSynchronize());

                    // Check if any updates occurred (light edges)
                    CUDA_CHECK(cudaMemcpy(&h_updated, d_updated_flag, sizeof(int), cudaMemcpyDeviceToHost));

                    light_iterations++;

                    if (h_updated == 0)
                    {
                        break;
                    }
                }

                iterations++;

                if (config->verbose && iterations % 100 == 0)
                {
                    printf("[Delta-Stepping] Iteration %d, bucket %d\n", iterations, h_min_bucket);
                }
            }

            // Copy results back
            CUDA_CHECK(cudaMemcpy(labels_out, gpu_graph->labels,
                                  num_nodes * sizeof(int), cudaMemcpyDeviceToHost));

            // Count labeled nodes for debugging
            int labeled_count = 0;
            int label0_count = 0;
            int label1_count = 0;
            for (int i = 0; i < num_nodes; i++)
            {
                if (labels_out[i] >= 0)
                {
                    labeled_count++;
                    if (labels_out[i] == 0)
                        label0_count++;
                    else if (labels_out[i] == 1)
                        label1_count++;
                }
            }
            printf("[Delta-Stepping] Stats: iterations=%d, labeled=%d/%d (%.1f%%), label0=%d, label1=%d\n",
                   iterations, labeled_count, num_nodes, 100.0 * labeled_count / num_nodes,
                   label0_count, label1_count);

            // Cleanup
            cudaFree(d_seed_nodes);
            cudaFree(d_seed_labels);
            cudaFree(d_bucket_nodes);
            cudaFree(d_bucket_count);
            cudaFree(d_min_bucket);
            cudaFree(d_updated_flag);

            if (config->verbose)
            {
                printf("[Delta-Stepping] Completed in %d iterations\n", iterations);
            }

            return iterations;
        }

        // GPU kernel: Relaxation kernel with atomic cost+label update
        // Uses 64-bit atomic to update cost and label together
        // in_queue[] prevents adding the same node multiple times to the queue
        __global__ void oift_relax_kernel_v2(
            const int *__restrict__ scene_data,
            const int *__restrict__ frontier,
            int frontier_size,
            unsigned long long *__restrict__ cost_label, // Packed: cost (high 32) | label (low 32)
            int *__restrict__ in_queue,                  // Flag to prevent duplicate queue entries
            int *__restrict__ next_queue,
            int *__restrict__ next_queue_count,
            float polaridade,
            int nx, int ny, int nz)
        {
            // Shared memory for adjacency
            __shared__ int s_adj_dx[6];
            __shared__ int s_adj_dy[6];
            __shared__ int s_adj_dz[6];
            __shared__ int s_adj_size;

            if (threadIdx.x < 6)
            {
                s_adj_dx[threadIdx.x] = c_adj_dx[threadIdx.x];
                s_adj_dy[threadIdx.x] = c_adj_dy[threadIdx.x];
                s_adj_dz[threadIdx.x] = c_adj_dz[threadIdx.x];
            }
            if (threadIdx.x == 0)
            {
                s_adj_size = c_adj_size;
            }
            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= frontier_size)
                return;

            int p = frontier[tid];

            // Unpack cost and label for p
            // Use volatile to ensure we read the latest value
            unsigned long long packed_p = cost_label[p];
            int cost_p = (int)(packed_p >> 32);
            int label_p = (int)(packed_p & 0xFFFFFFFF);

            // Skip if this node hasn't been labeled yet (shouldn't happen, but safety check)
            if (label_p < 0)
                return;

            // Get coordinates
            int nxy = nx * ny;
            int pz = p / nxy;
            int rem = p % nxy;
            int py = rem / nx;
            int px = rem % nx;

            int Ip = scene_data[p];

            // Process all neighbors
            for (int i = 0; i < s_adj_size; i++)
            {
                int qx = px + s_adj_dx[i];
                int qy = py + s_adj_dy[i];
                int qz = pz + s_adj_dz[i];

                // Bounds check
                if (qx < 0 || qx >= nx || qy < 0 || qy >= ny || qz < 0 || qz >= nz)
                    continue;

                int q = qz * nxy + qy * nx + qx;

                int Iq = scene_data[q];

                // OIFT edge weight
                // NOTE: polaridade is already in [-1, 1] range (not multiplied by 100)
                int w = abs(Ip - Iq);
                float per_pq = (label_p > 0) ? polaridade : -polaridade;

                if (Ip > Iq)
                {
                    w = (int)((float)w * (1.0f + per_pq));
                }
                else if (Ip < Iq)
                {
                    w = (int)((float)w * (1.0f - per_pq));
                }

                // OIFT uses PATH-MAX: new_cost = MAX(cost[p], edge_weight)
                int new_cost = (cost_p > w) ? cost_p : w;

                // Pack new cost and label
                unsigned long long new_packed = (((unsigned long long)(unsigned int)new_cost) << 32) |
                                                ((unsigned int)label_p);

                // Atomic compare-and-swap loop to update cost+label together
                unsigned long long old_packed = cost_label[q];
                while (true)
                {
                    int old_cost = (int)(old_packed >> 32);

                    // Only update if new cost is STRICTLY better
                    // When costs are equal, first arrival wins (matches CPU FIFO behavior)
                    if (new_cost >= old_cost)
                        break;

                    unsigned long long result = atomicCAS(&cost_label[q], old_packed, new_packed);

                    if (result == old_packed)
                    {
                        // Successfully updated - add to queue only if not already queued
                        int was_queued = atomicExch(&in_queue[q], 1);
                        if (was_queued == 0)
                        {
                            int pos = atomicAdd(next_queue_count, 1);
                            next_queue[pos] = q;
                        }
                        break;
                    }

                    // CAS failed, retry with new value
                    old_packed = result;
                }
            }
        }

        // Kernel to clear in_queue flags for nodes in current frontier
        // This allows them to be added to the queue again in future iterations
        __global__ void oift_clear_queue_flags_kernel(
            const int *__restrict__ frontier,
            int frontier_size,
            int *__restrict__ in_queue)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= frontier_size)
                return;
            in_queue[frontier[tid]] = 0;
        }

        // Kernel to remove duplicates from frontier using a visited flag
        __global__ void oift_compact_frontier_kernel(
            const int *__restrict__ raw_frontier,
            int raw_size,
            int *__restrict__ visited,
            int *__restrict__ compact_frontier,
            int *__restrict__ compact_count)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= raw_size)
                return;

            int node = raw_frontier[tid];

            // Try to mark as visited (atomicExch returns old value)
            int was_visited = atomicExch(&visited[node], 1);

            if (was_visited == 0)
            {
                // First time seeing this node, add to compact frontier
                int pos = atomicAdd(compact_count, 1);
                compact_frontier[pos] = node;
            }
        }

        // Reset visited flags
        __global__ void oift_reset_visited_kernel(
            const int *__restrict__ frontier,
            int frontier_size,
            int *__restrict__ visited)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= frontier_size)
                return;
            visited[frontier[tid]] = 0;
        }

        void oift_gpu_delta_stepping(
            const int *scene_data,
            int nx, int ny, int nz,
            const int *seed_nodes,
            const int *seed_labels,
            int num_seeds,
            float polaridade,
            int *labels_out)
        {
            int num_nodes = nx * ny * nz;

            // Create 6-connected adjacency
            int adj_x[] = {-1, 1, 0, 0, 0, 0};
            int adj_y[] = {0, 0, -1, 1, 0, 0};
            int adj_z[] = {0, 0, 0, 0, -1, 1};
            int adj_size = 6;

            // Copy adjacency to constant memory
            CUDA_CHECK(cudaMemcpyToSymbol(c_adj_dx, adj_x, adj_size * sizeof(int)));
            CUDA_CHECK(cudaMemcpyToSymbol(c_adj_dy, adj_y, adj_size * sizeof(int)));
            CUDA_CHECK(cudaMemcpyToSymbol(c_adj_dz, adj_z, adj_size * sizeof(int)));
            CUDA_CHECK(cudaMemcpyToSymbol(c_adj_size, &adj_size, sizeof(int)));

            printf("[OIFT-GPU] Starting parallel OIFT (allow re-updates), nodes=%d, seeds=%d, pol=%.2f\n",
                   num_nodes, num_seeds, polaridade);

            // Allocate device memory
            // Use 64-bit packed cost+label to ensure atomic updates of both together
            // Format: high 32 bits = cost (unsigned), low 32 bits = label
            int *d_scene_data;
            unsigned long long *d_cost_label; // Packed: cost (high 32) | label (low 32)
            int *d_in_queue;                  // Flag to prevent duplicate queue entries
            int *d_frontier, *d_next_frontier;
            int *d_next_frontier_count;

            CUDA_CHECK(cudaMalloc(&d_scene_data, num_nodes * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_cost_label, num_nodes * sizeof(unsigned long long)));
            CUDA_CHECK(cudaMalloc(&d_in_queue, num_nodes * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_frontier, num_nodes * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_next_frontier, num_nodes * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_next_frontier_count, sizeof(int)));

            // Upload scene data
            CUDA_CHECK(cudaMemcpy(d_scene_data, scene_data, num_nodes * sizeof(int), cudaMemcpyHostToDevice));

            // Initialize arrays
            // Pack: cost=INT_MAX (as unsigned), label=-1 (as unsigned = 0xFFFFFFFF)
            // INT_MAX = 0x7FFFFFFF, so packed = 0x7FFFFFFF_FFFFFFFF
            unsigned long long init_packed = ((unsigned long long)0x7FFFFFFFU << 32) | 0xFFFFFFFFU;
            thrust::device_ptr<unsigned long long> cost_label_ptr(d_cost_label);
            thrust::device_ptr<int> in_queue_ptr(d_in_queue);

            thrust::fill(cost_label_ptr, cost_label_ptr + num_nodes, init_packed);
            thrust::fill(in_queue_ptr, in_queue_ptr + num_nodes, 0);

            // Initialize seeds: cost=0, label=seed_label, mark as in_queue
            std::vector<int> h_frontier(num_seeds);
            std::vector<int> h_in_queue_seeds(num_seeds, 1); // All seeds start in queue
            for (int i = 0; i < num_seeds; i++)
            {
                int node = seed_nodes[i];
                int lbl = seed_labels[i];
                // Pack cost=0, label=lbl
                unsigned long long seed_packed = ((unsigned long long)0U << 32) | ((unsigned int)lbl);
                CUDA_CHECK(cudaMemcpy(&d_cost_label[node], &seed_packed, sizeof(unsigned long long), cudaMemcpyHostToDevice));
                // Mark as in_queue
                int one = 1;
                CUDA_CHECK(cudaMemcpy(&d_in_queue[node], &one, sizeof(int), cudaMemcpyHostToDevice));
                h_frontier[i] = node;
            }

            // Upload initial frontier (seeds)
            CUDA_CHECK(cudaMemcpy(d_frontier, h_frontier.data(), num_seeds * sizeof(int), cudaMemcpyHostToDevice));
            int h_frontier_count = num_seeds;

            int iteration = 0;

            while (h_frontier_count > 0)
            {
                // Reset next frontier count
                int zero = 0;
                CUDA_CHECK(cudaMemcpy(d_next_frontier_count, &zero, sizeof(int), cudaMemcpyHostToDevice));

                // Process current frontier - relax all neighbors
                int grid_size = (h_frontier_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                oift_relax_kernel_v2<<<grid_size, BLOCK_SIZE>>>(
                    d_scene_data, d_frontier, h_frontier_count,
                    d_cost_label, d_in_queue,
                    d_next_frontier, d_next_frontier_count,
                    polaridade, nx, ny, nz);

                CUDA_CHECK(cudaDeviceSynchronize());

                // Clear in_queue flags for processed nodes AFTER processing
                // This allows them to be re-added in future iterations if cost improves
                int grid_clear = (h_frontier_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                oift_clear_queue_flags_kernel<<<grid_clear, BLOCK_SIZE>>>(
                    d_frontier, h_frontier_count, d_in_queue);

                CUDA_CHECK(cudaDeviceSynchronize());

                // Get next frontier size
                CUDA_CHECK(cudaMemcpy(&h_frontier_count, d_next_frontier_count, sizeof(int), cudaMemcpyDeviceToHost));

                // Swap frontiers
                std::swap(d_frontier, d_next_frontier);

                iteration++;

                if (iteration % 50 == 0)
                {
                    printf("[OIFT-GPU] Iteration %d, queue size=%d\n", iteration, h_frontier_count);
                }
            }

            // Extract labels from packed cost_label
            std::vector<unsigned long long> h_cost_label(num_nodes);
            CUDA_CHECK(cudaMemcpy(h_cost_label.data(), d_cost_label, num_nodes * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

            // Count results and extract labels
            int labeled_count = 0, label0_count = 0, label1_count = 0;
            for (int i = 0; i < num_nodes; i++)
            {
                int label = (int)(h_cost_label[i] & 0xFFFFFFFF);
                labels_out[i] = label;
                if (label >= 0)
                {
                    labeled_count++;
                    if (label == 0)
                        label0_count++;
                    else if (label == 1)
                        label1_count++;
                }
            }

            printf("[OIFT-GPU] Completed: iterations=%d, labeled=%d/%d (%.1f%%), label0=%d, label1=%d\n",
                   iteration, labeled_count, num_nodes, 100.0 * labeled_count / num_nodes,
                   label0_count, label1_count);

            // Cleanup
            cudaFree(d_scene_data);
            cudaFree(d_cost_label);
            cudaFree(d_in_queue);
            cudaFree(d_frontier);
            cudaFree(d_next_frontier);
            cudaFree(d_next_frontier_count);
        }

        // Pack cost (high 32 bits) and label (low 32 bits) into 64-bit value
        // This allows atomic update of both together
        __device__ __forceinline__ unsigned long long pack_cost_label(int cost, int label)
        {
            return (((unsigned long long)(unsigned int)cost) << 32) | ((unsigned int)label);
        }

        __device__ __forceinline__ void unpack_cost_label(unsigned long long packed, int *cost, int *label)
        {
            *cost = (int)(packed >> 32);
            *label = (int)(packed & 0xFFFFFFFF);
        }

        // Bellman-Ford kernel for OIFT with atomic cost+label update
        __global__ void oift_bellman_ford_kernel(
            const int *scene_data,
            unsigned long long *cost_label, // Packed cost+label
            int *updated,
            float polaridade,
            int nx, int ny, int nz,
            int num_nodes)
        {
            int p = blockIdx.x * blockDim.x + threadIdx.x;
            if (p >= num_nodes)
                return;

            unsigned long long packed_p = cost_label[p];
            int cost_p, label_p;
            unpack_cost_label(packed_p, &cost_p, &label_p);

            // Only process nodes that have been labeled (reached by some seed)
            if (label_p < 0)
                return;

            // Get coordinates
            int nxy = nx * ny;
            int pz = p / nxy;
            int rem = p % nxy;
            int py = rem / nx;
            int px = rem % nx;

            int Ip = scene_data[p];

            // Try to relax all neighbors
            for (int i = 0; i < c_adj_size; i++)
            {
                int qx = px + c_adj_dx[i];
                int qy = py + c_adj_dy[i];
                int qz = pz + c_adj_dz[i];

                // Bounds check
                if (qx < 0 || qx >= nx || qy < 0 || qy >= ny || qz < 0 || qz >= nz)
                    continue;

                int q = qz * nxy + qy * nx + qx;
                int Iq = scene_data[q];

                // OIFT edge weight (matching CPU implementation exactly)
                // w = |I(p) - I(q)|
                // per_pq = (label > 0) ? +pol : -pol
                // if I(p) > I(q): w *= (1 + per_pq)
                // if I(p) < I(q): w *= (1 - per_pq)
                int w_base = abs(Ip - Iq);
                float per_pq = (label_p > 0) ? polaridade : -polaridade;

                float w_float;
                if (Ip > Iq)
                {
                    w_float = w_base * (1.0f + per_pq);
                }
                else if (Ip < Iq)
                {
                    w_float = w_base * (1.0f - per_pq);
                }
                else
                {
                    w_float = 0.0f;
                }

                int new_cost = (int)(w_float + 0.5f); // Round to int

                // Atomic compare-and-swap to update cost+label together
                unsigned long long new_packed = pack_cost_label(new_cost, label_p);
                unsigned long long old_packed = cost_label[q];

                while (true)
                {
                    int old_cost, old_label;
                    unpack_cost_label(old_packed, &old_cost, &old_label);

                    // Only update if new cost is strictly lower
                    if (new_cost >= old_cost)
                        break;

                    // Try to atomically update
                    unsigned long long result = atomicCAS(&cost_label[q], old_packed, new_packed);

                    if (result == old_packed)
                    {
                        // Successfully updated
                        atomicExch(updated, 1);
                        break;
                    }

                    // Another thread updated it, retry with new value
                    old_packed = result;
                }
            }
        }

        void print_gpu_memory_usage()
        {
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            printf("GPU Memory: %.2f MB used / %.2f MB total (%.2f%% free)\n",
                   (total_mem - free_mem) / (1024.0 * 1024.0),
                   total_mem / (1024.0 * 1024.0),
                   100.0 * free_mem / total_mem);
        }

        void gpu_warmup()
        {
            // Launch a tiny kernel to initialize GPU
            int *d_dummy;
            cudaMalloc(&d_dummy, sizeof(int));
            cudaFree(d_dummy);
            cudaDeviceSynchronize();
        }

    } // namespace gpu
} // namespace gft
