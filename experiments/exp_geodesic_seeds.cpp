// exp_geodesic_seeds.cpp
// Experiment 5: Geodesic distance-aware boundary seed placement
// Replaces uniform-stride boundary seed placement with distance-aware placement:
// - Compute EDT from all foreground seeds via BFS
// - Only place boundary seeds where EDT exceeds threshold (boundary_stride * 2)
// - Denser seeds near foreground (within boundary_stride distance)

#include "gft.h"

#include <algorithm>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <queue>
#include <cmath>
#include <climits>

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

// ==================== GEODESIC EDT-BASED BOUNDARY SEEDS ====================

// Compute approximate BFS-based distance transform from foreground seed positions.
// Returns array of size label->n with distances (in voxel steps, 0 = seed location).
// Uses 6-connected BFS (Manhattan distance approximation).
static std::vector<int> compute_bfs_edt(gft::sScene32 *label, const int *S)
{
    int n = label->n;
    std::vector<int> dist(n, INT_MAX);
    std::queue<int> bfsq;

    // Initialize: foreground seeds (label > 0) as source
    if (S != NULL)
    {
        for (int i = 1; i <= S[0]; i++)
        {
            int p = S[i];
            if (p >= 0 && p < n && label->data[p] > 0)
            {
                dist[p] = 0;
                bfsq.push(p);
            }
        }
    }

    // 6-connected neighbor offsets for BFS
    // We'll use a simple face adjacency
    gft::sAdjRel3 *A6 = gft::AdjRel3::Spheric(1.0f);

    gft::Voxel u, v;
    while (!bfsq.empty())
    {
        int p = bfsq.front();
        bfsq.pop();

        u.c.x = gft::Scene32::GetAddressX(label, p);
        u.c.y = gft::Scene32::GetAddressY(label, p);
        u.c.z = gft::Scene32::GetAddressZ(label, p);

        for (int i = 1; i < A6->n; i++)
        {
            v.v = u.v + A6->d[i].v;
            if (gft::Scene32::IsValidVoxel(label, v))
            {
                int q = gft::Scene32::GetVoxelAddress(label, v);
                if (dist[q] == INT_MAX)
                {
                    dist[q] = dist[p] + 1;
                    bfsq.push(q);
                }
            }
        }
    }

    gft::AdjRel3::Destroy(&A6);
    return dist;
}

// Geodesic-distance-aware boundary seed injection
// - boundary seeds on volume faces only placed where EDT > threshold
// - stride depends on distance: denser near seeds, sparser far away
int inject_boundary_seeds_geodesic(gft::sScene32 *label, int *&S,
                                    int current_count, int stride,
                                    const std::vector<int> &edt)
{
    int xsize = label->xsize;
    int ysize = label->ysize;
    int zsize = label->zsize;
    int threshold = stride * 2; // min EDT to place a boundary seed
    std::vector<int> boundary_positions;

    for (int z = 0; z < zsize; z++)
        for (int y = 0; y < ysize; y++)
            for (int x = 0; x < xsize; x++)
            {
                bool on_face = (x == 0 || x == xsize - 1 ||
                                y == 0 || y == ysize - 1 ||
                                z == 0 || z == zsize - 1);
                if (!on_face) continue;

                int p = gft::Scene32::GetVoxelAddress(label, x, y, z);
                if (label->data[p] != NIL) continue;

                int d = (edt[p] == INT_MAX) ? INT_MAX : edt[p];

                // Only place if far enough from foreground seeds
                if (d <= threshold) continue;

                // Adaptive stride based on distance:
                // - d <= stride:        dense (stride/2 step, but we already skipped these above)
                // - stride < d <= 3*stride: normal stride
                // - d > 3*stride:       sparse (2*stride step)
                int effective_stride;
                if (d <= stride)
                    effective_stride = MAX(1, stride / 2);
                else if (d <= 3 * stride)
                    effective_stride = stride;
                else
                    effective_stride = stride * 2;

                if ((x % effective_stride != 0) &&
                    (y % effective_stride != 0) &&
                    (z % effective_stride != 0))
                    continue;

                label->data[p] = 0;
                boundary_positions.push_back(p);
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

int main(int argc, char **argv)
{
    std::cout << "=== EXP 5: Geodesic distance-aware boundary seed placement ===" << std::endl;

    gft::sScene32 *scn, *fscn, *label;
    gft::sAdjRel3 *A;
    FILE *fp;
    int *S;
    int p, i, j, nseeds, x, y, z, id, lb, Imin;
    int niter = 50;
    float pol = 0.5f;
    int percentile = 50;
    int boundary_stride = 8;
    char *output_file;

    std::vector<float> per_class_vec;
    bool use_per_class = false;

    if (argc < 7)
    {
        fprintf(stdout, "usage:\n");
        fprintf(stdout, "exp_geodesic_seeds <volume> <file_seeds> <pol> <niter> <percentile> <output_file> [boundary_stride] [pol_file]\n");
        exit(0);
    }

    A = gft::AdjRel3::Spheric(1.0f);
    scn = gft::Scene32::Read(argv[1]);
    label = gft::Scene32::Create(scn);
    gft::Scene32::Fill(label, NIL);

    Imin = gft::Scene32::GetMinimumValue(scn);
    if (Imin < 0)
        for (p = 0; p < scn->n; p++)
            scn->data[p] += (-Imin);

    pol = atof(argv[3]);
    niter = atoi(argv[4]);
    percentile = atoi(argv[5]);
    output_file = argv[6];
    if (argc >= 8) boundary_stride = atoi(argv[7]);

    char *pol_file = NULL;
    if (argc >= 9) pol_file = argv[8];

    if (pol_file != NULL)
    {
        FILE *pfp = fopen(pol_file, "r");
        if (pfp != NULL)
        {
            int n_labels;
            if (fscanf(pfp, " %d", &n_labels) == 1 && n_labels > 0)
            {
                per_class_vec.resize(n_labels);
                bool ok = true;
                for (int ci = 0; ci < n_labels; ci++)
                {
                    float pv;
                    if (fscanf(pfp, " %f", &pv) != 1) { ok = false; break; }
                    per_class_vec[ci] = pv * 100.0f;
                }
                if (ok) use_per_class = true;
            }
            fclose(pfp);
        }
    }

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

    // Compute geodesic EDT from foreground seeds before injecting boundary seeds
    if (boundary_stride > 0)
    {
        DebugTimer::getInstance().startEvent("BFS EDT from foreground seeds");
        std::vector<int> edt = compute_bfs_edt(label, S);
        DebugTimer::getInstance().endEvent("BFS EDT from foreground seeds");

        DebugTimer::getInstance().startEvent("Geodesic boundary seed injection");
        int n_boundary = inject_boundary_seeds_geodesic(label, S, j, boundary_stride, edt);
        DebugTimer::getInstance().endEvent("Geodesic boundary seed injection");

        std::cout << "Geodesic boundary seeds: " << n_boundary
                  << " bg seeds added (stride=" << boundary_stride
                  << ", threshold=" << (boundary_stride * 2) << ")" << std::endl;
        std::cout << "Total seeds: " << S[0] << std::endl;
    }

    DebugTimer::getInstance().startEvent("GaussianBlur (1x)");
    fscn = gft::Scene32::GaussianBlur(scn);
    DebugTimer::getInstance().endEvent("GaussianBlur (1x)");

    DebugTimer::getInstance().startEvent("GaussianBlur (2x)");
    scn = fscn;
    fscn = gft::Scene32::GaussianBlur(scn);
    DebugTimer::getInstance().endEvent("GaussianBlur (2x)");
    gft::Scene32::Destroy(&scn);

    DebugTimer::getInstance().startEvent("OIFT_Multi");
    if (use_per_class)
    {
        int ml = (int)per_class_vec.size() - 1;
        gft::ift::OIFT_Multi_PerClass(A, fscn, per_class_vec.data(), ml, S, label);
    }
    else
    {
        gft::ift::OIFT_Multi(A, fscn, pol * 100.0f, S, label);
    }
    DebugTimer::getInstance().endEvent("OIFT_Multi");

    DebugTimer::getInstance().startEvent("ORelax_1_Multi");
    if (use_per_class)
    {
        int ml = (int)per_class_vec.size() - 1;
        gft::ift::ORelax_1_Multi_PerClass(A, fscn, per_class_vec.data(), ml, S, label, niter);
    }
    else
    {
        gft::ift::ORelax_1_Multi(A, fscn, pol * 100.0f, S, label, niter);
    }
    DebugTimer::getInstance().endEvent("ORelax_1_Multi");

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
