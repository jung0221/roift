/**
 * @file delta_stepping.cuh
 * @brief Delta-Stepping algorithm for parallel shortest path on GPU
 * 
 * Implements the Meyer & Sanders Delta-Stepping algorithm adapted for
 * the Oriented Image Foresting Transform (OIFT).
 * 
 * Reference: "Delta-Stepping: A Parallelizable Shortest Path Algorithm"
 *            Meyer & Sanders, ESA 2003
 */

#ifndef DELTA_STEPPING_CUH
#define DELTA_STEPPING_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <cfloat>
#include <cstdint>

namespace gft {
namespace gpu {

//==============================================================================
// Constants and Configuration
//==============================================================================

// Default delta value for bucket partitioning
#define DEFAULT_DELTA 100.0f

// Maximum number of iterations for convergence
#define MAX_DELTA_ITERATIONS 10000

// Thread block configuration
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Memory alignment for coalesced access
#define MEMORY_ALIGNMENT 128

//==============================================================================
// Data Structures
//==============================================================================

/**
 * @brief Graph structure for GPU processing
 */
struct GPUGraph {
    int num_nodes;           // Total number of nodes
    int num_edges;           // Total number of edges
    
    // CSR representation
    int* row_offsets;        // Size: num_nodes + 1
    int* col_indices;        // Size: num_edges
    float* edge_weights;     // Size: num_edges
    float* edge_orientations;// Size: num_edges (orientation component)
    
    // Node properties
    int* labels;             // Final label assignment
    float* costs;            // Path costs
    int* roots;              // Root node for each node
    int* predecessors;       // Predecessor in shortest path tree
    
    // Scene data for weight calculation
    int* scene_data;         // Original image intensities
    int nx, ny, nz;          // Volume dimensions
    
    // Adjacency offsets (for 3D navigation)
    int* adj_offsets_x;      // Adjacency dx values
    int* adj_offsets_y;      // Adjacency dy values  
    int* adj_offsets_z;      // Adjacency dz values
    int adj_size;            // Number of adjacency elements
};

/**
 * @brief Delta-Stepping bucket structure
 */
struct DeltaBuckets {
    int* bucket_indices;     // Which bucket each node belongs to
    int* bucket_starts;      // Start index for each bucket
    int* bucket_sizes;       // Number of nodes in each bucket
    int* node_order;         // Nodes sorted by bucket
    int num_buckets;         // Total number of buckets
    float delta;             // Delta value for bucket width
};

/**
 * @brief Seeds information
 */
struct GPUSeeds {
    int* seed_nodes;         // Node indices of seeds
    int* seed_labels;        // Labels for each seed (0 = background, 1 = object)
    int num_seeds;           // Total number of seeds
    int num_object_seeds;    // Number of object seeds
    int num_background_seeds;// Number of background seeds
};

/**
 * @brief Configuration for Delta-Stepping OIFT
 */
struct DeltaConfig {
    float delta;             // Bucket width
    float polaridade;        // Orientation parameter (-1 to 1)
    float Wmax;              // Maximum edge weight
    int max_iterations;      // Maximum iterations
    bool use_light_edges;    // Use light/heavy edge separation
    bool verbose;            // Print debug info
};

//==============================================================================
// GPU Kernels - Delta Stepping Core
//==============================================================================

/**
 * @brief Initialize costs and labels for seeds
 */
__global__ void init_seeds_kernel(
    float* costs,
    int* labels,
    int* roots,
    const int* seed_nodes,
    const int* seed_labels,
    int num_seeds,
    int num_nodes
);

/**
 * @brief Assign nodes to buckets based on current cost
 */
__global__ void assign_buckets_kernel(
    const float* costs,
    int* bucket_indices,
    const int* active_mask,
    float delta,
    int num_nodes
);

/**
 * @brief Process light edges within current bucket (relaxation)
 */
__global__ void process_light_edges_kernel(
    const int* bucket_nodes,
    int bucket_size,
    const int* row_offsets,
    const int* col_indices,
    const float* edge_weights,
    float* costs,
    int* labels,
    int* roots,
    int* updated_flags,
    float delta,
    float polaridade
);

/**
 * @brief Process heavy edges (edges crossing buckets)
 */
__global__ void process_heavy_edges_kernel(
    const int* bucket_nodes,
    int bucket_size,
    const int* row_offsets,
    const int* col_indices,
    const float* edge_weights,
    float* costs,
    int* labels,
    int* roots,
    int* updated_flags,
    float delta,
    float polaridade
);

/**
 * @brief Relaxation kernel for OIFT-style path computation
 */
__global__ void oift_relax_kernel(
    const int* active_nodes,
    int num_active,
    const int* row_offsets,
    const int* col_indices,
    const int* scene_data,
    float* costs,
    int* labels,
    int* roots,
    int* predecessors,
    int* updated_flags,
    float Wmax,
    float polaridade,
    int nx, int ny, int nz
);

/**
 * @brief Compute edge weight with orientation (OIFT formula)
 */
__device__ __forceinline__ float compute_oriented_weight(
    int Ip, int Iq, float Wmax, float polaridade, int label_q
) {
    float w = fabsf((float)(Ip - Iq));
    
    // Apply orientation based on label
    float per = (label_q == 0) ? polaridade : -polaridade;
    
    if (Ip > Iq) {
        w *= (1.0f + per);
    } else if (Ip < Iq) {
        w *= (1.0f - per);
    }
    
    return Wmax - w;
}

/**
 * @brief Find minimum cost in frontier (reduction)
 */
__global__ void find_min_cost_kernel(
    const float* costs,
    const int* frontier_nodes,
    int frontier_size,
    float* min_cost_out
);

/**
 * @brief Mark converged nodes (no more updates possible)
 */
__global__ void mark_converged_kernel(
    const float* costs,
    const float* old_costs,
    int* converged_flags,
    int num_nodes,
    float epsilon
);

//==============================================================================
// Host Functions
//==============================================================================

/**
 * @brief Allocate GPU graph structure
 */
GPUGraph* allocate_gpu_graph(int num_nodes, int num_edges, int adj_size);

/**
 * @brief Free GPU graph structure
 */
void free_gpu_graph(GPUGraph* graph);

/**
 * @brief Allocate delta buckets
 */
DeltaBuckets* allocate_delta_buckets(int num_nodes, float delta, float max_cost);

/**
 * @brief Free delta buckets
 */
void free_delta_buckets(DeltaBuckets* buckets);

/**
 * @brief Convert CPU adjacency to GPU format
 */
void convert_adjacency_to_gpu(
    GPUGraph* gpu_graph,
    const int* adj_x, const int* adj_y, const int* adj_z,
    int adj_size
);

/**
 * @brief Build CSR graph from 3D volume adjacency
 */
void build_csr_graph_3d(
    GPUGraph* gpu_graph,
    const int* scene_data,
    int nx, int ny, int nz,
    const int* adj_x, const int* adj_y, const int* adj_z,
    int adj_size,
    float Wmax,
    float polaridade
);

/**
 * @brief Main Delta-Stepping OIFT function
 * 
 * @param gpu_graph Pre-allocated GPU graph structure
 * @param seeds Seed information (object and background)
 * @param config Algorithm configuration
 * @param labels_out Output labels (host memory, pre-allocated)
 * @return Number of iterations performed
 */
int delta_stepping_oift(
    GPUGraph* gpu_graph,
    const GPUSeeds* seeds,
    const DeltaConfig* config,
    int* labels_out
);

/**
 * @brief Simplified interface for Delta-Stepping OIFT
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
    const int* scene_data,
    int nx, int ny, int nz,
    const int* seed_nodes,
    const int* seed_labels,
    int num_seeds,
    float polaridade,
    int* labels_out
);

//==============================================================================
// Utility Functions
//==============================================================================

/**
 * @brief Check CUDA error and print message
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * @brief Get optimal block/grid dimensions
 */
inline void get_launch_config(int num_elements, int& grid_size, int& block_size) {
    block_size = BLOCK_SIZE;
    grid_size = (num_elements + block_size - 1) / block_size;
}

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

#endif // DELTA_STEPPING_CUH
