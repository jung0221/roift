/**
 * @file oift_fmax_gpu.cu
 * @brief GPU OIFT with bottleneck (fmax) path cost.
 *
 * cost[q] = max(cost[p], w(p,q))  — monotonically non-decreasing along paths.
 * This makes iterative parallel relaxation correct and convergent.
 *
 * Active-frontier approach: only recently-updated voxels relax their neighbours.
 * Uses 64-bit atomicMin packing (cost << 32 | label) for race-free updates.
 *
 * Supports per-class polarity via device array.
 */

#include <cuda_runtime.h>
#include <climits>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// 6-connected offsets
__constant__ int DX[6] = { 1, -1,  0,  0,  0,  0};
__constant__ int DY[6] = { 0,  0,  1, -1,  0,  0};
__constant__ int DZ[6] = { 0,  0,  0,  0,  1, -1};

#define COST_INF 0x7FFFFFFF
#define PACK(cost, label) (((unsigned long long)(cost) << 32) | (unsigned long long)(unsigned int)(label))
#define UNPACK_COST(cl)   ((int)((cl) >> 32))
#define UNPACK_LABEL(cl)  ((int)((cl) & 0xFFFFFFFF))

// ── Frontier relaxation kernel ──────────────────────────────────────────────

__global__ void kernel_fmax_relax(
    const int* __restrict__          image,
    unsigned long long* __restrict__ cost_label,    // packed (cost, label)
    const float* __restrict__        per_class,     // per-class polarity (×100)
    int max_label,
    int xsize, int ysize, int zsize,
    const int* __restrict__ frontier_in,
    int n_frontier,
    int* __restrict__ frontier_out,
    int* __restrict__ n_frontier_out,
    int* __restrict__ in_frontier)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_frontier) return;

    int idx = frontier_in[tid];
    unsigned long long my_cl = cost_label[idx];
    int my_cost  = UNPACK_COST(my_cl);
    int my_label = UNPACK_LABEL(my_cl);

    // Per-class polarity
    float per_pq = 0.0f;
    if (my_label >= 0 && my_label <= max_label)
        per_pq = per_class[my_label];

    int xy = xsize * ysize;
    int z = idx / xy;
    int rem = idx - z * xy;
    int y = rem / xsize;
    int x = rem - y * xsize;

    #pragma unroll
    for (int i = 0; i < 6; i++) {
        int nx = x + DX[i];
        int ny = y + DY[i];
        int nz = z + DZ[i];

        if (nx < 0 || nx >= xsize ||
            ny < 0 || ny >= ysize ||
            nz < 0 || nz >= zsize) continue;

        int nidx = nz * xy + ny * xsize + nx;

        // Edge weight with polarity (same formula as CPU OIFT_Multi)
        int Ip = image[idx];
        int Iq = image[nidx];
        float base = (float)abs(Ip - Iq);
        float wf;
        if (Ip > Iq)
            wf = base * (1.0f + per_pq / 100.0f);
        else if (Ip < Iq)
            wf = base * (1.0f - per_pq / 100.0f);
        else
            wf = base;
        if (wf < 0.0f) wf = 0.0f;
        int w = (int)wf;

        // fmax path cost: max(my_cost, edge_weight)
        int proposed = max(my_cost, w);

        // Pack and atomicMin — wins if (proposed < old_cost) or
        // (proposed == old_cost && my_label < old_label)
        unsigned long long proposed_cl = PACK(proposed, my_label);
        unsigned long long old = atomicMin(&cost_label[nidx], proposed_cl);

        if (proposed_cl < old) {
            // We improved this neighbour — add to next frontier (once)
            int was = atomicExch(&in_frontier[nidx], 1);
            if (was == 0) {
                int pos = atomicAdd(n_frontier_out, 1);
                frontier_out[pos] = nidx;
            }
        }
    }
}

// ── Clear frontier flags kernel ─────────────────────────────────────────────

__global__ void kernel_clear_flags(
    int* __restrict__ in_frontier,
    const int* __restrict__ frontier,
    int n_frontier)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_frontier) return;
    in_frontier[frontier[tid]] = 0;
}

// ── Host API ────────────────────────────────────────────────────────────────

extern "C" {

/**
 * Run GPU fmax-OIFT.
 *
 * @param h_image     Host int32 image (n voxels, already shifted to non-negative)
 * @param h_labels    Host int32 label map (NIL=-1 for unlabelled, 0=bg, 1..N=fg).
 *                    On return, filled with the segmentation result.
 * @param n           Total voxels
 * @param xsize,ysize,zsize  Volume dimensions
 * @param h_per_class Host float array of per-class polarity (×100), length max_label+1
 * @param max_label   Maximum label value
 * @param h_seeds     Host int32 seed voxel indices
 * @param n_seeds     Number of seeds
 * @return            Number of iterations, or -1 on error
 */
int oift_fmax_gpu_run(
    const int*   h_image,
    int*         h_labels,
    int n,
    int xsize, int ysize, int zsize,
    const float* h_per_class,
    int max_label,
    const int*   h_seeds,
    int n_seeds)
{
    // Device allocations
    int*                d_image;
    unsigned long long* d_cost_label;
    float*              d_per_class;
    int*                d_frontier_a;
    int*                d_frontier_b;
    int*                d_n_frontier;
    int*                d_in_frontier;

    size_t sz_int  = (size_t)n * sizeof(int);
    size_t sz_ull  = (size_t)n * sizeof(unsigned long long);
    size_t sz_pc   = (size_t)(max_label + 1) * sizeof(float);

    cudaMalloc(&d_image,       sz_int);
    cudaMalloc(&d_cost_label,  sz_ull);
    cudaMalloc(&d_per_class,   sz_pc);
    cudaMalloc(&d_frontier_a,  sz_int);
    cudaMalloc(&d_frontier_b,  sz_int);
    cudaMalloc(&d_n_frontier,  sizeof(int));
    cudaMalloc(&d_in_frontier, sz_int);

    // Upload image and per-class polarity
    cudaMemcpy(d_image,     h_image,     sz_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_per_class, h_per_class, sz_pc,  cudaMemcpyHostToDevice);

    // Initialize cost_label: seeds = (0, label), rest = (COST_INF, 0)
    std::vector<unsigned long long> h_cl(n, PACK(COST_INF, 0));
    std::vector<int> h_frontier;
    h_frontier.reserve(n_seeds);

    for (int i = 0; i < n_seeds; i++) {
        int p  = h_seeds[i];
        int lb = h_labels[p];
        if (lb < 0) lb = 0;  // map -1 → 0 (bg)
        h_cl[p] = PACK(0, lb);
        h_frontier.push_back(p);
    }

    cudaMemcpy(d_cost_label, h_cl.data(), sz_ull, cudaMemcpyHostToDevice);

    // Initial frontier
    int h_frontier_size = (int)h_frontier.size();
    cudaMemcpy(d_frontier_a, h_frontier.data(), h_frontier_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_in_frontier, 0, sz_int);

    // Iterative relaxation
    int threads = 256;
    int total_iters = 0;
    int max_iters = 2 * (xsize + ysize + zsize);  // generous diameter bound

    int* d_frontier_in  = d_frontier_a;
    int* d_frontier_out = d_frontier_b;

    while (h_frontier_size > 0 && total_iters < max_iters) {
        // Reset output counter
        int zero = 0;
        cudaMemcpy(d_n_frontier, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // Clear in_frontier flags for the CURRENT output frontier
        // (they were set in the previous iteration's relax kernel)
        // On first iteration, in_frontier is already zero from memset above.
        // We clear flags corresponding to frontier_out entries from last time.
        // Since frontier_in now points to last iteration's output, clear those.
        if (total_iters > 0) {
            int cblocks = (h_frontier_size + threads - 1) / threads;
            kernel_clear_flags<<<cblocks, threads>>>(
                d_in_frontier, d_frontier_in, h_frontier_size);
        }

        // Relax
        int blocks = (h_frontier_size + threads - 1) / threads;
        kernel_fmax_relax<<<blocks, threads>>>(
            d_image, d_cost_label, d_per_class, max_label,
            xsize, ysize, zsize,
            d_frontier_in, h_frontier_size,
            d_frontier_out, d_n_frontier, d_in_frontier);
        cudaDeviceSynchronize();

        total_iters++;
        cudaMemcpy(&h_frontier_size, d_n_frontier, sizeof(int), cudaMemcpyDeviceToHost);

        // Swap
        int* tmp = d_frontier_in;
        d_frontier_in  = d_frontier_out;
        d_frontier_out = tmp;

        // Progress
        if (total_iters <= 5 || total_iters % 100 == 0)
            printf("  iter %d: frontier=%d\n", total_iters, h_frontier_size);
    }

    if (total_iters >= max_iters)
        printf("  WARNING: hit max iterations (%d), remaining frontier=%d\n",
               max_iters, h_frontier_size);

    // Download and extract labels
    cudaMemcpy(h_cl.data(), d_cost_label, sz_ull, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        h_labels[i] = UNPACK_LABEL(h_cl[i]);

    // Cleanup
    cudaFree(d_image);
    cudaFree(d_cost_label);
    cudaFree(d_per_class);
    cudaFree(d_frontier_a);
    cudaFree(d_frontier_b);
    cudaFree(d_n_frontier);
    cudaFree(d_in_frontier);

    return total_iters;
}

} // extern "C"
