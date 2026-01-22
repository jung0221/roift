/**
 * @file benchmark_gpu.cpp
 * @brief Benchmark comparing CPU vs GPU implementations of ROIFT
 *
 * Tests:
 * 1. CPU OIFT + CPU ORelax (baseline)
 * 2. CPU OIFT + GPU ORelax (hybrid - recommended)
 * 3. GPU Delta-Stepping + GPU ORelax (full GPU - experimental)
 */

#include "gft.h"
#include "delta_stepping.h" // C++ wrapper (not .cuh)
#include "orelax_gpu.h"     // C++ wrapper (not .cuh)

#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cstring>

using namespace std::chrono;

// Forward declarations
void run_cpu_pipeline(gft::sScene32 *scn, int *S, float pol, int niter, gft::sScene32 *label);
void run_hybrid_pipeline(gft::sScene32 *scn, int *S, float pol, int niter, gft::sScene32 *label);
void run_gpu_pipeline(gft::sScene32 *scn, int *S, float pol, int niter, gft::sScene32 *label);

// ==================== CPU Implementation ====================
void ORelax_CPU(
    gft::sAdjRel3 *A,
    gft::sScene32 *scn,
    float per,
    int *S,
    gft::sScene32 *label,
    int niter)
{
    gft::ift::ORelax_1(A, scn, per, S, label, niter);
}

// ==================== GPU ORelax Implementation ====================
void ORelax_GPU(
    gft::sAdjRel3 *A,
    gft::sScene32 *scn,
    float per,
    int *S,
    gft::sScene32 *label,
    int niter)
{
    int n = scn->n;
    int nx = scn->xsize;
    int ny = scn->ysize;
    int nz = scn->zsize;

    // Compute Wmax
    float Wmax = 0;
    for (int p = 0; p < n; p++)
    {
        if (label->data[p] != NIL)
        {
            for (int i = 1; i < A->n; i++)
            {
                int qx = gft::Scene32::GetAddressX(scn, p) + A->d[i].axis.x;
                int qy = gft::Scene32::GetAddressY(scn, p) + A->d[i].axis.y;
                int qz = gft::Scene32::GetAddressZ(scn, p) + A->d[i].axis.z;
                if (gft::Scene32::IsValidVoxel(scn, qx, qy, qz))
                {
                    int q = gft::Scene32::GetVoxelAddress(scn, qx, qy, qz);
                    float w = std::abs(scn->data[p] - scn->data[q]);
                    if (w > Wmax)
                        Wmax = w;
                }
            }
        }
    }
    Wmax *= 2.0f;

    // Build mask nodes
    std::vector<int> mask_nodes;
    for (int p = 0; p < n; p++)
    {
        if (label->data[p] != NIL)
        {
            mask_nodes.push_back(p);
        }
    }

    // Build seed arrays
    std::vector<int> seed_nodes;
    std::vector<float> seed_values;
    for (int i = 1; i <= S[0]; i++)
    {
        int p = S[i];
        seed_nodes.push_back(p);
        seed_values.push_back(label->data[p] == 1 ? 1.0f : 0.0f);
    }

    // Initialize float labels
    std::vector<float> flabel(n, 0.5f);
    for (int p = 0; p < n; p++)
    {
        if (label->data[p] != NIL)
        {
            flabel[p] = (label->data[p] == 1) ? 1.0f : 0.0f;
        }
    }

    // Build adjacency
    std::vector<int> adj_dx(A->n), adj_dy(A->n), adj_dz(A->n);
    for (int i = 0; i < A->n; i++)
    {
        adj_dx[i] = A->d[i].axis.x;
        adj_dy[i] = A->d[i].axis.y;
        adj_dz[i] = A->d[i].axis.z;
    }

    // GPU params
    gft::gpu::ORelaxParams params;
    params.Wmax = Wmax;
    params.percentile = per;
    params.num_iterations = niter;
    params.nx = nx;
    params.ny = ny;
    params.nz = nz;
    params.adj_size = A->n;

    std::vector<float> flabel_out(n);

    gft::gpu::orelax_gpu_simple(
        scn->data,
        mask_nodes.data(),
        (int)mask_nodes.size(),
        flabel.data(),
        flabel_out.data(),
        n,
        seed_nodes.data(),
        seed_values.data(),
        (int)seed_nodes.size(),
        adj_dx.data(),
        adj_dy.data(),
        adj_dz.data(),
        A->n,
        &params);

    for (int p = 0; p < n; p++)
    {
        if (label->data[p] != NIL)
        {
            label->data[p] = (flabel_out[p] >= 0.5f) ? 1 : 0;
        }
    }
}

// ==================== MAIN ====================
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: benchmark_gpu <volume.nii> <seeds.txt> [niter=50] [pol=0.5]" << std::endl;
        return 1;
    }

    const char *volume_path = argv[1];
    const char *seeds_path = argv[2];
    int niter = (argc > 3) ? atoi(argv[3]) : 50;
    float pol = (argc > 4) ? atof(argv[4]) : 0.5f;

    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "GPU ROIFT BENCHMARK" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Check GPU
    if (!gft::gpu::check_gpu_available())
    {
        std::cout << "ERROR: No CUDA-capable GPU found!" << std::endl;
        return 1;
    }
    gft::gpu::print_gpu_info();
    gft::gpu::gpu_warmup();

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Volume: " << volume_path << std::endl;
    std::cout << "  Seeds: " << seeds_path << std::endl;
    std::cout << "  Iterations: " << niter << std::endl;
    std::cout << "  Polarity: " << pol << std::endl;
    std::cout << std::endl;

    // Load volume
    gft::sScene32 *scn_original = gft::Scene32::Read((char *)volume_path);
    if (!scn_original)
    {
        std::cout << "ERROR: Cannot read volume" << std::endl;
        return 1;
    }

    // Normalize intensities
    int Imin = gft::Scene32::GetMinimumValue(scn_original);
    if (Imin < 0)
    {
        for (int p = 0; p < scn_original->n; p++)
            scn_original->data[p] += (-Imin);
    }

    // Apply Gaussian blur
    gft::sScene32 *scn_blur1 = gft::Scene32::GaussianBlur(scn_original);
    gft::sScene32 *scn = gft::Scene32::GaussianBlur(scn_blur1);
    gft::Scene32::Destroy(&scn_blur1);
    gft::Scene32::Destroy(&scn_original);

    // Read seeds
    FILE *fp = fopen(seeds_path, "r");
    if (!fp)
    {
        std::cout << "ERROR: Cannot read seeds" << std::endl;
        return 1;
    }

    int nseeds;
    fscanf(fp, " %d", &nseeds);
    int *S = (int *)calloc(nseeds + 1, sizeof(int));
    std::vector<int> seed_labels_original(nseeds);

    int j = 0;
    for (int i = 0; i < nseeds; i++)
    {
        int x, y, z, id, lb;
        fscanf(fp, " %d %d %d %d %d", &x, &y, &z, &id, &lb);
        if (gft::Scene32::IsValidVoxel(scn, x, y, z))
        {
            int p = gft::Scene32::GetVoxelAddress(scn, x, y, z);
            j++;
            S[j] = p;
            seed_labels_original[j - 1] = lb;
        }
    }
    S[0] = j;
    fclose(fp);

    gft::sAdjRel3 *A = gft::AdjRel3::Spheric(1.0);

    std::cout << "Volume size: " << scn->xsize << " x " << scn->ysize << " x " << scn->zsize << std::endl;
    std::cout << "Total voxels: " << scn->n << std::endl;
    std::cout << "Seeds: " << S[0] << std::endl;
    std::cout << std::endl;

    // ===== BENCHMARK 1: CPU Only =====
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "TEST 1: CPU OIFT + CPU ORelax (baseline)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    gft::sScene32 *label_cpu = gft::Scene32::Create(scn);
    gft::Scene32::Fill(label_cpu, NIL);
    for (int i = 1; i <= S[0]; i++)
    {
        label_cpu->data[S[i]] = seed_labels_original[i - 1];
    }

    auto t1_start = high_resolution_clock::now();
    gft::ift::OIFT(A, scn, pol * 100.0, S, label_cpu);
    auto t1_oift = high_resolution_clock::now();
    ORelax_CPU(A, scn, pol * 100.0, S, label_cpu, niter);
    auto t1_end = high_resolution_clock::now();

    double cpu_oift_ms = duration_cast<microseconds>(t1_oift - t1_start).count() / 1000.0;
    double cpu_orelax_ms = duration_cast<microseconds>(t1_end - t1_oift).count() / 1000.0;
    double cpu_total_ms = cpu_oift_ms + cpu_orelax_ms;

    std::cout << "  OIFT:   " << std::fixed << std::setprecision(2) << cpu_oift_ms << " ms" << std::endl;
    std::cout << "  ORelax: " << std::fixed << std::setprecision(2) << cpu_orelax_ms << " ms" << std::endl;
    std::cout << "  TOTAL:  " << std::fixed << std::setprecision(2) << cpu_total_ms << " ms" << std::endl;

    // ===== BENCHMARK 2: Hybrid (CPU OIFT + GPU ORelax) =====
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "TEST 2: CPU OIFT + GPU ORelax (hybrid - RECOMMENDED)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    gft::sScene32 *label_hybrid = gft::Scene32::Create(scn);
    gft::Scene32::Fill(label_hybrid, NIL);
    for (int i = 1; i <= S[0]; i++)
    {
        label_hybrid->data[S[i]] = seed_labels_original[i - 1];
    }

    auto t2_start = high_resolution_clock::now();
    gft::ift::OIFT(A, scn, pol * 100.0, S, label_hybrid);
    auto t2_oift = high_resolution_clock::now();
    ORelax_GPU(A, scn, pol * 100.0, S, label_hybrid, niter);
    auto t2_end = high_resolution_clock::now();

    double hybrid_oift_ms = duration_cast<microseconds>(t2_oift - t2_start).count() / 1000.0;
    double hybrid_orelax_ms = duration_cast<microseconds>(t2_end - t2_oift).count() / 1000.0;
    double hybrid_total_ms = hybrid_oift_ms + hybrid_orelax_ms;

    std::cout << "  OIFT:   " << std::fixed << std::setprecision(2) << hybrid_oift_ms << " ms" << std::endl;
    std::cout << "  ORelax: " << std::fixed << std::setprecision(2) << hybrid_orelax_ms << " ms (GPU)" << std::endl;
    std::cout << "  TOTAL:  " << std::fixed << std::setprecision(2) << hybrid_total_ms << " ms" << std::endl;

    // ===== BENCHMARK 3: Full GPU (Delta-Stepping + GPU ORelax) =====
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "TEST 3: GPU Delta-Stepping + GPU ORelax (experimental)" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    gft::sScene32 *label_gpu = gft::Scene32::Create(scn);
    gft::Scene32::Fill(label_gpu, NIL);
    for (int i = 1; i <= S[0]; i++)
    {
        label_gpu->data[S[i]] = seed_labels_original[i - 1];
    }

    std::vector<int> seed_nodes;
    std::vector<int> seed_labels;
    for (int i = 1; i <= S[0]; i++)
    {
        seed_nodes.push_back(S[i]);
        seed_labels.push_back(seed_labels_original[i - 1]);
    }
    std::vector<int> labels_out(scn->n);

    auto t3_start = high_resolution_clock::now();
    gft::gpu::oift_gpu_delta_stepping(
        scn->data,
        scn->xsize, scn->ysize, scn->zsize,
        seed_nodes.data(),
        seed_labels.data(),
        (int)seed_nodes.size(),
        pol,
        labels_out.data());
    auto t3_oift = high_resolution_clock::now();

    // Copy delta-stepping result
    for (int p = 0; p < scn->n; p++)
    {
        if (labels_out[p] >= 0)
        {
            label_gpu->data[p] = labels_out[p];
        }
    }

    ORelax_GPU(A, scn, pol * 100.0, S, label_gpu, niter);
    auto t3_end = high_resolution_clock::now();

    double gpu_oift_ms = duration_cast<microseconds>(t3_oift - t3_start).count() / 1000.0;
    double gpu_orelax_ms = duration_cast<microseconds>(t3_end - t3_oift).count() / 1000.0;
    double gpu_total_ms = gpu_oift_ms + gpu_orelax_ms;

    std::cout << "  OIFT:   " << std::fixed << std::setprecision(2) << gpu_oift_ms << " ms (Delta-Stepping)" << std::endl;
    std::cout << "  ORelax: " << std::fixed << std::setprecision(2) << gpu_orelax_ms << " ms (GPU)" << std::endl;
    std::cout << "  TOTAL:  " << std::fixed << std::setprecision(2) << gpu_total_ms << " ms" << std::endl;

    // ===== RESULTS SUMMARY =====
    std::cout << "\n"
              << std::string(70, '=') << std::endl;
    std::cout << "SPEEDUP SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    std::cout << std::left << std::setw(35) << "Configuration"
              << std::right << std::setw(15) << "Time (ms)"
              << std::right << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    std::cout << std::left << std::setw(35) << "CPU Only (baseline)"
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << cpu_total_ms
              << std::right << std::setw(15) << "1.00x" << std::endl;

    std::cout << std::left << std::setw(35) << "Hybrid (CPU OIFT + GPU ORelax)"
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << hybrid_total_ms
              << std::right << std::setw(15) << std::fixed << std::setprecision(2)
              << (cpu_total_ms / hybrid_total_ms) << "x" << std::endl;

    std::cout << std::left << std::setw(35) << "Full GPU (Delta + GPU ORelax)"
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << gpu_total_ms
              << std::right << std::setw(15) << std::fixed << std::setprecision(2)
              << (cpu_total_ms / gpu_total_ms) << "x" << std::endl;

    std::cout << std::string(70, '=') << std::endl;

    // ORelax-specific speedup
    std::cout << "\nORelax Speedup: " << std::fixed << std::setprecision(2)
              << (cpu_orelax_ms / hybrid_orelax_ms) << "x" << std::endl;

    // Compare results (Dice coefficient approximation)
    int match_hybrid = 0, match_gpu = 0;
    int total_labeled = 0;
    for (int p = 0; p < scn->n; p++)
    {
        if (label_cpu->data[p] != NIL)
        {
            total_labeled++;
            if (label_cpu->data[p] == label_hybrid->data[p])
                match_hybrid++;
            if (label_cpu->data[p] == label_gpu->data[p])
                match_gpu++;
        }
    }

    std::cout << "\nResult Agreement vs CPU baseline:" << std::endl;
    std::cout << "  Hybrid: " << std::fixed << std::setprecision(2)
              << (100.0 * match_hybrid / total_labeled) << "%" << std::endl;
    std::cout << "  Full GPU: " << std::fixed << std::setprecision(2)
              << (100.0 * match_gpu / total_labeled) << "%" << std::endl;

    // Cleanup
    gft::Scene32::Destroy(&scn);
    gft::Scene32::Destroy(&label_cpu);
    gft::Scene32::Destroy(&label_hybrid);
    gft::Scene32::Destroy(&label_gpu);
    gft::AdjRel3::Destroy(&A);
    free(S);

    std::cout << "\nBenchmark complete!" << std::endl;

    return 0;
}
