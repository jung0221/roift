/**
 * @file oiftrelax_gpu.cpp
 * @brief GPU-accelerated ROIFT with Delta-Stepping OIFT and ORelax
 *
 * This version uses CUDA for:
 * - Gaussian Blur (parallel convolution)
 * - Delta-Stepping for OIFT (parallel shortest path)
 * - ORelax GPU (parallel relaxation - main speedup)
 *
 * Expected speedup: 10-30x vs CPU single-thread
 */

#include "gft.h"
#include "delta_stepping.h"    // C++ wrapper (not .cuh)
#include "orelax_gpu.h"        // C++ wrapper (not .cuh)
#include "gaussian_blur_gpu.h" // C++ wrapper for GPU blur

#include <algorithm>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstring>

// ==================== DEBUG/TIMING UTILITIES ====================
class DebugTimer
{
public:
    struct Event
    {
        std::string name;
        std::chrono::high_resolution_clock::time_point start_time;
        double elapsed_ms;
    };

    static DebugTimer &getInstance()
    {
        static DebugTimer instance;
        return instance;
    }

    void startEvent(const std::string &name)
    {
        Event evt;
        evt.name = name;
        evt.start_time = std::chrono::high_resolution_clock::now();
        evt.elapsed_ms = 0.0;
        events.push_back(evt);

#ifdef _DEBUG
        std::cout << "[DEBUG] >>> START: " << name << std::endl;
#endif
    }

    void endEvent(const std::string &name)
    {
        auto now = std::chrono::high_resolution_clock::now();
        for (auto &evt : events)
        {
            if (evt.name == name && evt.elapsed_ms == 0.0)
            {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - evt.start_time);
                evt.elapsed_ms = duration.count() / 1000.0;
#ifdef _DEBUG
                std::cout << "[DEBUG] <<< END: " << name << " | Elapsed: "
                          << std::fixed << std::setprecision(2) << evt.elapsed_ms << " ms" << std::endl;
#endif
                return;
            }
        }
    }

    void printSummary()
    {
        std::cout << "\n"
                  << std::string(60, '=') << std::endl;
        std::cout << "GPU TIMING SUMMARY" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        double total = 0.0;
        for (const auto &evt : events)
        {
            if (evt.elapsed_ms > 0.0)
            {
                std::cout << std::left << std::setw(40) << evt.name
                          << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                          << evt.elapsed_ms << " ms" << std::endl;
                total += evt.elapsed_ms;
            }
        }
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::left << std::setw(40) << "TOTAL"
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                  << total << " ms" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }

private:
    std::vector<Event> events;
};
// ================================================================

gft::sScene32 *get_dilation_border(gft::sScene32 *scn, float radius_sphere)
{
    gft::sScene32 *dil = NULL;
    int p, q, n, i;
    gft::Voxel u, v;
    gft::sAdjRel3 *A = gft::AdjRel3::Spheric(radius_sphere);
    n = scn->n;
    dil = gft::Scene32::Create(scn);
    gft::Scene32::Fill(dil, 0);
    for (p = 0; p < n; p++)
    {
        if (scn->data[p] == 1)
        {
            u.c.x = gft::Scene32::GetAddressX(scn, p);
            u.c.y = gft::Scene32::GetAddressY(scn, p);
            u.c.z = gft::Scene32::GetAddressZ(scn, p);
            for (i = 0; i < A->n; i++)
            {
                v.v = u.v + A->d[i].v;
                if (gft::Scene32::IsValidVoxel(scn, v))
                {
                    q = gft::Scene32::GetVoxelAddress(scn, v);
                    if (scn->data[q] == 0)
                    {
                        dil->data[q] = 1;
                    }
                }
            }
        }
    }
    gft::AdjRel3::Destroy(&A);
    return dil;
}

struct value_position
{
    int value;
    int position;
    value_position(int value, int position) : value(value), position(position) {}
};

void condition_percentil(gft::sScene32 *scn, gft::sScene32 *border, float percentil)
{
    int p, q, n;
    std::vector<value_position> border_values;
    n = scn->n;
    for (p = 0; p < n; p++)
    {
        if (border->data[p] == 1)
            border_values.emplace_back(scn->data[p], p);
    }

    // Check if we have any border values
    if (border_values.empty())
    {
        return;
    }

    std::stable_sort(border_values.begin(), border_values.end(), [](const value_position &a, const value_position &b)
                     { return a.value < b.value; });
    float index_percentile = (percentil / 100.0) * (border_values.size() - 1);
    int value_position = static_cast<int>(index_percentile);
    int num_voxels_border = border_values.size();

    // Ensure we don't access out of bounds
    for (p = value_position + 1; p < num_voxels_border && p < (int)border_values.size(); p++)
    {
        q = border_values[p].position;
        if (q >= 0 && q < n)
        {
            border->data[q] = 0;
        }
    }
}

void dilation_conditional(gft::sScene32 *scn, gft::sScene32 *label, float radius_sphere, float percentile)
{
    gft::sScene32 *dilation_border;
    int p, n;

    dilation_border = get_dilation_border(label, radius_sphere);
    condition_percentil(scn, dilation_border, percentile);

    n = scn->n;
    for (p = 0; p < n; p++)
    {
        if (dilation_border->data[p] == 1)
            label->data[p] = 1;
    }

    gft::Scene32::Destroy(&dilation_border);
}

// ==================== GPU-ACCELERATED ORelax ====================
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
    Wmax *= 2.0f; // Safety margin

    // Build mask nodes (labeled nodes)
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

    // Build adjacency arrays
    std::vector<int> adj_dx(A->n), adj_dy(A->n), adj_dz(A->n);
    for (int i = 0; i < A->n; i++)
    {
        adj_dx[i] = A->d[i].axis.x;
        adj_dy[i] = A->d[i].axis.y;
        adj_dz[i] = A->d[i].axis.z;
    }

    // Setup GPU parameters
    gft::gpu::ORelaxParams params;
    params.Wmax = Wmax;
    params.percentile = per;
    params.num_iterations = niter;
    params.nx = nx;
    params.ny = ny;
    params.nz = nz;
    params.adj_size = A->n;

    // Output buffer
    std::vector<float> flabel_out(n);

    // Run GPU ORelax
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

    // Convert back to integer labels
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
    gft::sScene32 *scn, *fscn, *label;
    gft::sAdjRel3 *A;
    FILE *fp;
    int *S;
    int p, i, j, nseeds, x, y, z, id, lb, Imin;
    int niter = 50;
    float pol = 0.5;
    int percentile = 50;
    char *output_file;
    bool use_gpu_oift = false;  // Delta-Stepping (experimental)
    bool use_gpu_orelax = true; // ORelax GPU (stable)

    if (argc < 7)
    {
        fprintf(stdout, "usage:\n");
        fprintf(stdout, "oiftrelax_gpu <volume> <file_seeds> <pol> <niter> <percentile> <output_file> [--delta]\n");
        fprintf(stdout, "\t pol......... Boundary polarity. Range: [-1.0, 1.0]\n");
        fprintf(stdout, "\t niter....... Number of iterations for relaxation\n");
        fprintf(stdout, "\t percentile.. Percentile for conditional dilation\n");
        fprintf(stdout, "\t output_file. Output label file (e.g., label.nii.gz)\n");
        fprintf(stdout, "\t --delta..... Use Delta-Stepping for OIFT (experimental)\n");
        exit(0);
    }

    // Check for optional flags
    for (int arg = 7; arg < argc; arg++)
    {
        if (strcmp(argv[arg], "--delta") == 0)
        {
            use_gpu_oift = true;
        }
        if (strcmp(argv[arg], "--cpu") == 0)
        {
            use_gpu_orelax = false;
        }
    }

    // Print GPU info
    std::cout << "\n=== GPU ROIFT - Delta Stepping ===" << std::endl;
    if (gft::gpu::check_gpu_available())
    {
        gft::gpu::print_gpu_info();
        gft::gpu::gpu_warmup();
    }
    else
    {
        std::cout << "WARNING: No CUDA-capable GPU found. Falling back to CPU." << std::endl;
        use_gpu_oift = false;
        use_gpu_orelax = false;
    }
    std::cout << std::endl;
    std::cout.flush();

    std::cout << "Reading volume: " << argv[1] << std::endl;
    std::cout.flush();

    A = gft::AdjRel3::Spheric(1.0);
    scn = gft::Scene32::Read(argv[1]);

    if (scn == NULL)
    {
        std::cerr << "ERROR: Could not read volume file: " << argv[1] << std::endl;
        return 1;
    }

    if (scn->n <= 0)
    {
        std::cerr << "ERROR: Invalid volume - no voxels" << std::endl;
        return 1;
    }

    std::cout << "Volume size: " << scn->xsize << " x " << scn->ysize << " x " << scn->zsize
              << " (" << scn->n << " voxels)" << std::endl;
    std::cout.flush();

    label = gft::Scene32::Create(scn);
    if (label == NULL)
    {
        std::cerr << "ERROR: Could not create label image" << std::endl;
        return 1;
    }
    gft::Scene32::Fill(label, NIL);

    Imin = gft::Scene32::GetMinimumValue(scn);
    if (Imin < 0)
    {
        for (p = 0; p < scn->n; p++)
            scn->data[p] += (-Imin);
    }

    pol = atof(argv[3]);
    niter = atoi(argv[4]);
    percentile = atoi(argv[5]);
    output_file = argv[6];

    std::cout << "Output: " << output_file << std::endl;
    std::cout << "Polarity: " << pol << std::endl;
    std::cout << "Iterations: " << niter << std::endl;
    std::cout << "Percentile: " << percentile << std::endl;
    std::cout << "Using GPU OIFT (Delta-Stepping): " << (use_gpu_oift ? "YES" : "NO (CPU)") << std::endl;
    std::cout << "Using GPU ORelax: " << (use_gpu_orelax ? "YES" : "NO (CPU)") << std::endl;
    std::cout << std::endl;
    std::cout.flush();

    // Read seeds
    std::cout << "Reading seeds: " << argv[2] << std::endl;
    std::cout.flush();

    fp = fopen(argv[2], "r");
    if (fp == NULL)
    {
        printf("Error reading seeds.\n");
        exit(1);
    }
    fscanf(fp, " %d", &nseeds);
    S = (int *)calloc((nseeds + 1), sizeof(int));
    S[0] = nseeds;
    j = 0;
    int max_seed_label = 0;
    bool corrected_negative_label = false;
    for (i = 0; i < nseeds; i++)
    {
        fscanf(fp, " %d %d %d %d %d", &x, &y, &z, &id, &lb);
        if (lb < 0)
        {
            lb = 0;
            corrected_negative_label = true;
        }
        if (gft::Scene32::IsValidVoxel(scn, x, y, z))
        {
            p = gft::Scene32::GetVoxelAddress(scn, x, y, z);
            j++;
            S[j] = p;
            label->data[p] = lb;
            if (lb > max_seed_label)
                max_seed_label = lb;
        }
    }
    S[0] = j;
    fclose(fp);

    if (j == 0)
    {
        std::cerr << "ERROR: No valid seeds found" << std::endl;
        return 1;
    }

    if (corrected_negative_label)
    {
        std::cout << "Warning: negative seed labels were clamped to 0 (background)." << std::endl;
    }

    const bool multilabel_mode = (max_seed_label > 1);
    if (multilabel_mode)
    {
        use_gpu_oift = false;
        use_gpu_orelax = false;
        std::cout << "Multi-label seeds detected (max label=" << max_seed_label
                  << "). Switching to CPU OIFT_Multi + ORelax_1_Multi for correctness." << std::endl;
    }

    std::cout << "Seeds loaded: " << j << " valid seeds" << std::endl;
    std::cout.flush();

    // ===== PROCESSING PIPELINE =====
    std::cout << "\n=== Starting Processing Pipeline ===" << std::endl;
    std::cout.flush();

    // Gaussian Blur 2x using GPU
    DebugTimer::getInstance().startEvent("GaussianBlur (2x) [GPU]");
    fscn = gft::Scene32::Create(scn);
    gft::gpu::gaussian_blur_gpu_2x(
        scn->data,
        fscn->data,
        scn->xsize, scn->ysize, scn->zsize);
    fscn->dx = scn->dx;
    fscn->dy = scn->dy;
    fscn->dz = scn->dz;
    DebugTimer::getInstance().endEvent("GaussianBlur (2x) [GPU]");

    gft::Scene32::Destroy(&scn);

    // OIFT - CPU version (Delta-Stepping experimental)
    if (use_gpu_oift)
    {
        DebugTimer::getInstance().startEvent("OIFT (Delta-Stepping GPU)");

        // Build seed arrays for GPU
        std::vector<int> seed_nodes;
        std::vector<int> seed_labels;
        for (int i = 1; i <= S[0]; i++)
        {
            seed_nodes.push_back(S[i]);
            seed_labels.push_back(label->data[S[i]]);
        }

        // Allocate output
        std::vector<int> labels_out(fscn->n);

        // Run Delta-Stepping OIFT
        // Note: CPU uses pol * 100 internally and divides by 100, so we pass pol directly
        // The GPU kernel expects polaridade in [-1, 1] range
        gft::gpu::oift_gpu_delta_stepping(
            fscn->data,
            fscn->xsize, fscn->ysize, fscn->zsize,
            seed_nodes.data(),
            seed_labels.data(),
            (int)seed_nodes.size(),
            pol, // polaridade in [-1, 1] range
            labels_out.data());

        // Copy back to label
        for (p = 0; p < fscn->n; p++)
        {
            if (labels_out[p] >= 0)
            {
                label->data[p] = labels_out[p];
            }
        }

        DebugTimer::getInstance().endEvent("OIFT (Delta-Stepping GPU)");
    }
    else
    {
        DebugTimer::getInstance().startEvent("OIFT_Multi (CPU)");
        gft::ift::OIFT_Multi(A, fscn, pol * 100.0, S, label);
        DebugTimer::getInstance().endEvent("OIFT_Multi (CPU)");
    }

    // ORelax - GPU version (main speedup)
    if (use_gpu_orelax)
    {
        DebugTimer::getInstance().startEvent("ORelax_1 (GPU - " + std::to_string(niter) + " iterations)");
        ORelax_GPU(A, fscn, pol * 100.0, S, label, niter);
        DebugTimer::getInstance().endEvent("ORelax_1 (GPU - " + std::to_string(niter) + " iterations)");
    }
    else
    {
        DebugTimer::getInstance().startEvent("ORelax_1_Multi (CPU - " + std::to_string(niter) + " iterations)");
        gft::ift::ORelax_1_Multi(A, fscn, pol * 100.0, S, label, niter);
        DebugTimer::getInstance().endEvent("ORelax_1_Multi (CPU - " + std::to_string(niter) + " iterations)");
    }

    int max_label = gft::Scene32::GetMaximumValue(label);
    if (max_label <= 1)
    {
        DebugTimer::getInstance().startEvent("Dilation Conditional [CPU]");
        dilation_conditional(fscn, label, 1, percentile);
        DebugTimer::getInstance().endEvent("Dilation Conditional [CPU]");
    }
    else
    {
        std::cout << "Skipping binary dilation post-process for multi-label result (max label="
                  << max_label << ")." << std::endl;
    }

    gft::Scene32::Destroy(&fscn);

    // Print timing summary
    DebugTimer::getInstance().printSummary();

    // Save result
    std::cout << "Saving result to: " << output_file << std::endl;
    std::cout.flush();

    gft::Scene32::Write(label, output_file);

    std::cout << "Result saved successfully!" << std::endl;

    // Cleanup
    free(S);
    gft::Scene32::Destroy(&label);
    gft::AdjRel3::Destroy(&A);

    return 0;
}
