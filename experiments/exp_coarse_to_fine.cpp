// exp_coarse_to_fine.cpp
// Experiment 3: Coarse-to-fine cascade
// 1. Downsample volume 2x
// 2. Run OIFT+ORelax on coarse volume
// 3. Upscale labels to full resolution (nearest neighbor)
// 4. Use upscaled labels as seeds for second OIFT+ORelax pass

#include "gft.h"

#include <algorithm>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <queue>

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
    }

    void endEvent(const std::string &name)
    {
        auto now = std::chrono::high_resolution_clock::now();
        for (auto &evt : events)
        {
            if (evt.name == name && evt.elapsed_ms == 0.0)
            {
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - evt.start_time);
                evt.elapsed_ms = duration.count();
                return;
            }
        }
    }

    void printSummary()
    {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "TIMING SUMMARY" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        double total = 0.0;
        for (const auto &evt : events)
        {
            if (evt.elapsed_ms > 0.0)
            {
                std::cout << std::left << std::setw(35) << evt.name
                          << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                          << evt.elapsed_ms << " ms" << std::endl;
                total += evt.elapsed_ms;
            }
        }
        std::cout << std::string(60, '-') << std::endl;
        std::cout << std::left << std::setw(35) << "TOTAL"
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
                        dil->data[q] = 1;
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
        if (border->data[p] == 1)
            border_values.emplace_back(scn->data[p], p);
    std::stable_sort(border_values.begin(), border_values.end(),
                     [](const value_position &a, const value_position &b){ return a.value < b.value; });
    float index_percentile = (percentil / 100.0f) * (border_values.size() - 1);
    int vp = static_cast<int>(index_percentile);
    int num_voxels_border = border_values.size();
    for (p = vp + 1; p < num_voxels_border; p++)
    {
        q = border_values[p].position;
        border->data[q] = 0;
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
        if (dilation_border->data[p] == 1)
            label->data[p] = 1;
    gft::Scene32::Destroy(&dilation_border);
}

int inject_boundary_seeds(gft::sScene32 *label, int *&S, int current_count, int stride)
{
    int xsize = label->xsize;
    int ysize = label->ysize;
    int zsize = label->zsize;
    std::vector<int> boundary_positions;
    for (int z = 0; z < zsize; z += stride)
        for (int y = 0; y < ysize; y += stride)
            for (int x = 0; x < xsize; x += stride)
            {
                bool on_face = (x == 0 || x == xsize - 1 ||
                                y == 0 || y == ysize - 1 ||
                                z == 0 || z == zsize - 1);
                if (!on_face) continue;
                int p = gft::Scene32::GetVoxelAddress(label, x, y, z);
                if (label->data[p] == NIL)
                {
                    label->data[p] = 0;
                    boundary_positions.push_back(p);
                }
            }
    int n_new = boundary_positions.size();
    if (n_new > 0)
    {
        int new_total = current_count + n_new;
        int *S_new = (int *)calloc((new_total + 1), sizeof(int));
        S_new[0] = new_total;
        for (int i = 1; i <= current_count; i++)
            S_new[i] = S[i];
        for (int i = 0; i < n_new; i++)
            S_new[current_count + 1 + i] = boundary_positions[i];
        free(S);
        S = S_new;
    }
    return n_new;
}

// Build a new seed array from label image (all labeled voxels become seeds)
// Returns newly allocated S array (must be freed by caller)
static int* build_seeds_from_label(gft::sScene32 *label, int *n_seeds_out)
{
    int n = label->n;
    int count = 0;
    for (int p = 0; p < n; p++)
        if (label->data[p] != NIL)
            count++;
    int *S = (int *)calloc(count + 1, sizeof(int));
    S[0] = count;
    int j = 1;
    for (int p = 0; p < n; p++)
        if (label->data[p] != NIL)
            S[j++] = p;
    if (n_seeds_out) *n_seeds_out = count;
    return S;
}

// Mark boundary voxels (having a neighbor with different label) as NIL
// so OIFT can refine them in the fine pass
static void mark_boundary_voxels_nil(gft::sScene32 *label, gft::sAdjRel3 *A)
{
    int n = label->n;
    gft::sScene32 *is_boundary = gft::Scene32::Create(label);
    gft::Scene32::Fill(is_boundary, 0);

    gft::Voxel u, v;
    for (int p = 0; p < n; p++)
    {
        if (label->data[p] == NIL) continue;
        u.c.x = gft::Scene32::GetAddressX(label, p);
        u.c.y = gft::Scene32::GetAddressY(label, p);
        u.c.z = gft::Scene32::GetAddressZ(label, p);
        for (int i = 1; i < A->n; i++)
        {
            v.v = u.v + A->d[i].v;
            if (gft::Scene32::IsValidVoxel(label, v))
            {
                int q = gft::Scene32::GetVoxelAddress(label, v);
                if (label->data[q] != label->data[p])
                {
                    is_boundary->data[p] = 1;
                    break;
                }
            }
        }
    }

    for (int p = 0; p < n; p++)
        if (is_boundary->data[p] == 1)
            label->data[p] = NIL;

    gft::Scene32::Destroy(&is_boundary);
}

int main(int argc, char **argv)
{
    std::cout << "=== EXP 3: Coarse-to-fine cascade ===" << std::endl;

    gft::sScene32 *scn, *fscn, *label;
    gft::sScene32 *scn_coarse, *label_coarse, *label_upscaled;
    gft::sAdjRel3 *A;
    FILE *fp;
    int *S, *S_coarse, *S_fine;
    int p, i, j, nseeds, x, y, z, id, lb, Imin;
    int niter = 50;
    float pol = 0.5f;
    int percentile = 50;
    int boundary_stride = 8;
    char *output_file;

    if (argc < 7)
    {
        fprintf(stdout, "usage:\n");
        fprintf(stdout, "exp_coarse_to_fine <volume> <file_seeds> <pol> <niter> <percentile> <output_file> [boundary_stride] [pol_file]\n");
        exit(0);
    }

    pol = atof(argv[3]);
    niter = atoi(argv[4]);
    percentile = atoi(argv[5]);
    output_file = argv[6];
    if (argc >= 8) boundary_stride = atoi(argv[7]);

    A = gft::AdjRel3::Spheric(1.0f);
    scn = gft::Scene32::Read(argv[1]);
    label = gft::Scene32::Create(scn);
    gft::Scene32::Fill(label, NIL);

    Imin = gft::Scene32::GetMinimumValue(scn);
    if (Imin < 0)
        for (p = 0; p < scn->n; p++)
            scn->data[p] += (-Imin);

    fp = fopen(argv[2], "r");
    if (!fp) { printf("Error reading seeds.\n"); exit(1); }
    fscanf(fp, " %d", &nseeds);
    S = (int *)calloc((nseeds + 1), sizeof(int));
    S[0] = nseeds;
    j = 0;
    for (i = 0; i < nseeds; i++)
    {
        fscanf(fp, " %d %d %d %d %d", &x, &y, &z, &id, &lb);
        if (lb < 0) lb = 0;
        if (gft::Scene32::IsValidVoxel(scn, x, y, z))
        {
            p = gft::Scene32::GetVoxelAddress(scn, x, y, z);
            j++;
            S[j] = p;
            label->data[p] = lb;
        }
    }
    S[0] = j;
    fclose(fp);

    if (boundary_stride > 0)
    {
        int n_boundary = inject_boundary_seeds(label, S, j, boundary_stride);
        std::cout << "Boundary seeds: " << n_boundary << " bg seeds added (stride=" << boundary_stride << ")" << std::endl;
    }

    DebugTimer::getInstance().startEvent("GaussianBlur (1x)");
    fscn = gft::Scene32::GaussianBlur(scn);
    DebugTimer::getInstance().endEvent("GaussianBlur (1x)");

    DebugTimer::getInstance().startEvent("GaussianBlur (2x)");
    scn = fscn;
    fscn = gft::Scene32::GaussianBlur(scn);
    DebugTimer::getInstance().endEvent("GaussianBlur (2x)");
    gft::Scene32::Destroy(&scn);

    // ========== COARSE PASS ==========
    DebugTimer::getInstance().startEvent("Subsampling (2x downsample)");
    scn_coarse = gft::Scene32::Subsampling(fscn);
    DebugTimer::getInstance().endEvent("Subsampling (2x downsample)");

    // Map original seeds to coarse coordinates (divide by 2)
    label_coarse = gft::Scene32::Create(scn_coarse);
    gft::Scene32::Fill(label_coarse, NIL);

    S_coarse = (int *)calloc(S[0] + 1, sizeof(int));
    S_coarse[0] = 0;
    {
        // Use the original label image to find seed positions and labels
        int n_orig = label->n;
        for (int si = 1; si <= S[0]; si++)
        {
            int pp = S[si];
            if (pp < 0 || pp >= n_orig) continue;
            int cx = gft::Scene32::GetAddressX(label, pp) / 2;
            int cy = gft::Scene32::GetAddressY(label, pp) / 2;
            int cz = gft::Scene32::GetAddressZ(label, pp) / 2;
            if (gft::Scene32::IsValidVoxel(scn_coarse, cx, cy, cz))
            {
                int cq = gft::Scene32::GetVoxelAddress(scn_coarse, cx, cy, cz);
                if (label_coarse->data[cq] == NIL)
                {
                    label_coarse->data[cq] = label->data[pp];
                    S_coarse[0]++;
                    S_coarse[S_coarse[0]] = cq;
                }
            }
        }
    }
    std::cout << "Coarse seeds: " << S_coarse[0] << std::endl;

    // Run coarse OIFT + ORelax
    DebugTimer::getInstance().startEvent("OIFT_Multi (coarse)");
    gft::ift::OIFT_Multi(A, scn_coarse, pol * 100.0f, S_coarse, label_coarse);
    DebugTimer::getInstance().endEvent("OIFT_Multi (coarse)");

    DebugTimer::getInstance().startEvent("ORelax_1_Multi (coarse)");
    gft::ift::ORelax_1_Multi(A, scn_coarse, pol * 100.0f, S_coarse, label_coarse, niter);
    DebugTimer::getInstance().endEvent("ORelax_1_Multi (coarse)");

    // Upscale coarse labels to full resolution
    DebugTimer::getInstance().startEvent("ScaleLabel (upsample)");
    label_upscaled = gft::Scene32::ScaleLabel(label_coarse, fscn, gft::none);
    DebugTimer::getInstance().endEvent("ScaleLabel (upsample)");

    gft::Scene32::Destroy(&scn_coarse);
    gft::Scene32::Destroy(&label_coarse);
    free(S_coarse);

    // ========== FINE PASS ==========
    // Mark boundary voxels of the upscaled labels as NIL (to be re-segmented)
    DebugTimer::getInstance().startEvent("Mark boundary voxels as NIL");
    mark_boundary_voxels_nil(label_upscaled, A);
    DebugTimer::getInstance().endEvent("Mark boundary voxels as NIL");

    // Build fine seeds: original seeds + interior voxels from upscaled labels
    // First, inject upscaled labels into the fine label image
    {
        int n_fine = label->n;
        for (int pp = 0; pp < n_fine; pp++)
        {
            if (label_upscaled->data[pp] != NIL && label->data[pp] == NIL)
                label->data[pp] = label_upscaled->data[pp];
        }
    }
    gft::Scene32::Destroy(&label_upscaled);

    // Build full seed list from the merged label
    int n_fine_seeds = 0;
    S_fine = build_seeds_from_label(label, &n_fine_seeds);
    std::cout << "Fine seeds (after merge): " << n_fine_seeds << std::endl;

    DebugTimer::getInstance().startEvent("OIFT_Multi (fine)");
    gft::ift::OIFT_Multi(A, fscn, pol * 100.0f, S_fine, label);
    DebugTimer::getInstance().endEvent("OIFT_Multi (fine)");

    DebugTimer::getInstance().startEvent("ORelax_1_Multi (fine)");
    gft::ift::ORelax_1_Multi(A, fscn, pol * 100.0f, S_fine, label, niter);
    DebugTimer::getInstance().endEvent("ORelax_1_Multi (fine)");

    free(S_fine);

    int max_label = gft::Scene32::GetMaximumValue(label);
    if (max_label <= 1)
    {
        DebugTimer::getInstance().startEvent("Dilation Conditional");
        dilation_conditional(fscn, label, 1.0f, (float)percentile);
        DebugTimer::getInstance().endEvent("Dilation Conditional");
    }
    else
    {
        std::cout << "Skipping binary dilation for multi-label (max=" << max_label << ")" << std::endl;
    }

    gft::Scene32::Destroy(&fscn);

    DebugTimer::getInstance().printSummary();
    gft::Scene32::Write(label, output_file);

    free(S);
    gft::Scene32::Destroy(&label);
    gft::AdjRel3::Destroy(&A);
    return 0;
}
