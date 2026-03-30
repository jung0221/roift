/**
 * @file oiftrelax_gpu.cpp
 * @brief CLI for GPU seed-based segmentation (fmax / additive cost).
 *
 * Usage: oiftrelax_gpu <volume> <seeds> <pol> <niter> <percentile> <output>
 *            [boundary_stride] [pol_file] [cost_mode]
 *
 * cost_mode: 0 = fmax (bottleneck path, default), 1 = additive (shortest path)
 */

#include "gft.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>

// From oift_gpu.cu
extern "C" int oift_gpu_run(
    int cost_mode,
    const int* h_image, int* h_labels, int n,
    int xsize, int ysize, int zsize,
    const float* h_per_class, int max_label,
    const int* h_seeds, int n_seeds);

static int inject_boundary_seeds(gft::sScene32* label, int*& S, int current_count, int stride)
{
    int xsize = label->xsize, ysize = label->ysize, zsize = label->zsize;
    std::vector<int> bpos;
    for (int z = 0; z < zsize; z += stride)
        for (int y = 0; y < ysize; y += stride)
            for (int x = 0; x < xsize; x += stride) {
                bool on_face = (x==0||x==xsize-1||y==0||y==ysize-1||z==0||z==zsize-1);
                if (!on_face) continue;
                int p = gft::Scene32::GetVoxelAddress(label, x, y, z);
                if (label->data[p] == NIL) {
                    label->data[p] = 0;
                    bpos.push_back(p);
                }
            }
    int n_new = (int)bpos.size();
    if (n_new > 0) {
        int new_total = current_count + n_new;
        int* S_new = (int*)calloc(new_total + 1, sizeof(int));
        S_new[0] = new_total;
        for (int i = 1; i <= current_count; i++) S_new[i] = S[i];
        for (int i = 0; i < n_new; i++) S_new[current_count + 1 + i] = bpos[i];
        free(S);
        S = S_new;
    }
    return n_new;
}

int main(int argc, char** argv)
{
    if (argc < 7) {
        fprintf(stderr, "usage: oiftrelax_gpu <volume> <seeds> <pol> <niter> <percentile> <output> [boundary_stride] [pol_file] [cost_mode]\n");
        fprintf(stderr, "  cost_mode: 0 = fmax (default), 1 = additive\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d MB)\n", prop.name, (int)(prop.totalGlobalMem / (1024*1024)));

    gft::sScene32* scn = gft::Scene32::Read(argv[1]);
    int xsize = scn->xsize, ysize = scn->ysize, zsize = scn->zsize;
    int n = scn->n;
    printf("Volume: %d×%d×%d (%d voxels)\n", xsize, ysize, zsize, n);

    int Imin = gft::Scene32::GetMinimumValue(scn);
    if (Imin < 0)
        for (int p = 0; p < n; p++) scn->data[p] += (-Imin);

    float pol = atof(argv[3]);
    // niter = argv[4], percentile = argv[5] (unused for GPU methods)
    char* output_file = argv[6];
    int boundary_stride = (argc >= 8) ? atoi(argv[7]) : 8;
    char* pol_file = (argc >= 9) ? argv[8] : NULL;
    int cost_mode = (argc >= 10) ? atoi(argv[9]) : 0;

    const char* mode_names[] = {"fmax", "additive"};
    printf("Cost mode: %s\n", mode_names[cost_mode]);

    // Label map
    gft::sScene32* label = gft::Scene32::Create(scn);
    gft::Scene32::Fill(label, NIL);

    // Read seeds
    FILE* fp = fopen(argv[2], "r");
    if (!fp) { fprintf(stderr, "Error reading seeds\n"); return 1; }
    int nseeds;
    fscanf(fp, " %d", &nseeds);
    int* S = (int*)calloc(nseeds + 1, sizeof(int));
    S[0] = nseeds;
    int j = 0;
    for (int i = 0; i < nseeds; i++) {
        int x, y, z, id, lb;
        fscanf(fp, " %d %d %d %d %d", &x, &y, &z, &id, &lb);
        if (lb < 0) lb = 0;
        if (gft::Scene32::IsValidVoxel(scn, x, y, z)) {
            int p = gft::Scene32::GetVoxelAddress(scn, x, y, z);
            j++;
            S[j] = p;
            label->data[p] = lb;
        }
    }
    S[0] = j;
    fclose(fp);

    // Boundary seeds
    if (boundary_stride > 0) {
        int nb = inject_boundary_seeds(label, S, j, boundary_stride);
        printf("Boundary seeds: %d (stride=%d), total: %d\n", nb, boundary_stride, S[0]);
    }

    // Per-class polarity
    int max_label = 0;
    for (int i = 1; i <= S[0]; i++) {
        int lb = label->data[S[i]];
        if (lb > max_label) max_label = lb;
    }

    std::vector<float> per_class_vec(max_label + 1, 0.0f);
    bool use_per_class = false;

    if (pol_file && strlen(pol_file) > 0) {
        FILE* pfp = fopen(pol_file, "r");
        if (pfp) {
            int nl;
            if (fscanf(pfp, " %d", &nl) == 1 && nl > 0) {
                per_class_vec.resize(nl, 0.0f);
                max_label = nl - 1;
                bool ok = true;
                for (int i = 0; i < nl; i++) {
                    float pv;
                    if (fscanf(pfp, " %f", &pv) != 1) { ok = false; break; }
                    per_class_vec[i] = pv * 100.0f;
                }
                if (ok) use_per_class = true;
            }
            fclose(pfp);
        }
    }
    if (!use_per_class) {
        per_class_vec[0] = -pol * 100.0f;
        for (int c = 1; c <= max_label; c++)
            per_class_vec[c] = pol * 100.0f;
    }

    printf("Classes: %d  Polarity: %s\n", max_label,
           use_per_class ? "per-class" : (std::to_string(pol)).c_str());

    // Gaussian blur 2×
    auto t0 = std::chrono::high_resolution_clock::now();
    gft::sScene32* fscn = gft::Scene32::GaussianBlur(scn);
    gft::Scene32::Destroy(&scn);
    scn = fscn;
    fscn = gft::Scene32::GaussianBlur(scn);
    gft::Scene32::Destroy(&scn);
    scn = fscn;
    auto t1 = std::chrono::high_resolution_clock::now();
    double blur_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Gaussian Blur 2x (CPU): %.1f ms\n", blur_ms);

    // Seed index array
    if (S[0] == 0) {
        fprintf(stderr, "No valid seeds — writing empty output\n");
        gft::Scene32::Write(label, output_file);
        free(S);
        gft::Scene32::Destroy(&scn);
        gft::Scene32::Destroy(&label);
        return 0;
    }
    std::vector<int> seed_indices(S[0]);
    for (int i = 0; i < S[0]; i++)
        seed_indices[i] = S[i + 1];

    // Run GPU
    auto t2 = std::chrono::high_resolution_clock::now();
    int iters = oift_gpu_run(
        cost_mode,
        scn->data, label->data, n,
        xsize, ysize, zsize,
        per_class_vec.data(), max_label,
        seed_indices.data(), (int)seed_indices.size());
    auto t3 = std::chrono::high_resolution_clock::now();
    double oift_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    printf("GPU %s: %.1f ms, %d iterations\n", mode_names[cost_mode], oift_ms, iters);

    gft::Scene32::Destroy(&scn);
    gft::Scene32::Write(label, output_file);
    printf("Total: %.1f ms\n", blur_ms + oift_ms);

    free(S);
    gft::Scene32::Destroy(&label);
    return 0;
}
