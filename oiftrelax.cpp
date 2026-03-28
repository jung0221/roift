
#include "gft.h"

#include <algorithm>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
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
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - evt.start_time);
                evt.elapsed_ms = duration.count();
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
    int p, q, n, max, i;
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
    std::stable_sort(border_values.begin(), border_values.end(), [](const value_position &a, const value_position &b)
                     { return a.value < b.value; });
    float index_percentile = (percentil / 100.0) * (border_values.size() - 1);
    int value_position = static_cast<int>(index_percentile);
    int num_voxels_border = border_values.size();
    for (p = value_position + 1; p < num_voxels_border; p++)
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
    // gft::Scene32::Write(dilation_border, (char *)"label_dilation_border.nii.gz");
    condition_percentil(scn, dilation_border, percentile);
    // gft::Scene32::Write(dilation_border, (char *)"label_dilation_border_percentile.nii.gz");

    n = scn->n;
    for (p = 0; p < n; p++)
    {
        if (dilation_border->data[p] == 1)
            label->data[p] = 1;
    }

    gft::Scene32::Destroy(&dilation_border);
}

// ==================== BOUNDARY SEED INJECTION ====================
// Place background seeds (label=0) on all 6 volume faces to prevent
// foreground labels from leaking to the image boundary.
// Returns the number of boundary seeds added.
int inject_boundary_seeds(gft::sScene32 *label, int *&S, int current_count, int stride)
{
    int xsize = label->xsize;
    int ysize = label->ysize;
    int zsize = label->zsize;

    // Count how many boundary seeds we need (worst case: all face voxels)
    std::vector<int> boundary_positions;

    for (int z = 0; z < zsize; z += stride)
    {
        for (int y = 0; y < ysize; y += stride)
        {
            for (int x = 0; x < xsize; x += stride)
            {
                // Only process voxels on the 6 faces
                bool on_face = (x == 0 || x == xsize - 1 ||
                                y == 0 || y == ysize - 1 ||
                                z == 0 || z == zsize - 1);
                if (!on_face)
                    continue;

                int p = gft::Scene32::GetVoxelAddress(label, x, y, z);
                // Only add if no existing seed
                if (label->data[p] == NIL)
                {
                    label->data[p] = 0; // background
                    boundary_positions.push_back(p);
                }
            }
        }
    }

    // Reallocate S to fit new seeds
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
// ================================================================

int main(int argc, char **argv)
{
    gft::sScene32 *scn, *fscn, *label, *W, *Wx, *Wy, *Wz;

    gft::sAdjRel3 *A;
    clock_t end, start;
    double totaltime;
    FILE *fp;
    int *S;
    int p, i, j, nseeds, x, y, z, id, lb, Imin;
    int niter = 50;
    float pol = 0.5;
    int percentile = 50;
    int boundary_stride = 8;  // stride for boundary bg seeds (0 = disabled)
    char *output_file;
    if (argc < 7)
    {
        fprintf(stdout, "usage:\n");
        fprintf(stdout, "oiftrelax <volume> <file_seeds> <pol> <niter> <percentile> <output_file> [boundary_stride]\n");
        fprintf(stdout, "\t pol.............. Boundary polarity [-1.0, 1.0]\n");
        fprintf(stdout, "\t niter............ Relaxation iterations (0 = unlimited)\n");
        fprintf(stdout, "\t percentile...... Dilation percentile (binary mode only)\n");
        fprintf(stdout, "\t output_file..... Output label file (e.g., label.nii.gz)\n");
        fprintf(stdout, "\t boundary_stride. Stride for auto boundary bg seeds (default=8, 0=off)\n");
        exit(0);
    }

    A = gft::AdjRel3::Spheric(1.0);
    scn = gft::Scene32::Read(argv[1]);
    label = gft::Scene32::Create(scn);
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
    if (argc >= 8)
        boundary_stride = atoi(argv[7]);

    // printf("pol: %f, niter: %d\n", pol, niter);

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
        }
    }
    S[0] = j;
    fclose(fp);
    if (corrected_negative_label)
    {
        std::cout << "Warning: negative seed labels were clamped to 0 (background)." << std::endl;
    }

    // Inject background seeds on volume boundary faces
    int n_boundary = 0;
    if (boundary_stride > 0)
    {
        n_boundary = inject_boundary_seeds(label, S, j, boundary_stride);
        std::cout << "Boundary seeds: " << n_boundary << " bg seeds added (stride="
                  << boundary_stride << "), total seeds: " << S[0] << std::endl;
    }

    start = clock();

    DebugTimer::getInstance().startEvent("GaussianBlur (1x)");
    fscn = gft::Scene32::GaussianBlur(scn);
    DebugTimer::getInstance().endEvent("GaussianBlur (1x)");

    DebugTimer::getInstance().startEvent("GaussianBlur (2x)");
    scn = fscn;
    fscn = gft::Scene32::GaussianBlur(scn);
    DebugTimer::getInstance().endEvent("GaussianBlur (2x)");

    gft::Scene32::Destroy(&scn);

    DebugTimer::getInstance().startEvent("OIFT_Multi (Oriented Image Foresting Transform)");
    gft::ift::OIFT_Multi(A, fscn, pol * 100.0, S, label);
    DebugTimer::getInstance().endEvent("OIFT_Multi (Oriented Image Foresting Transform)");

    DebugTimer::getInstance().startEvent("ORelax_1_Multi (Relaxation - " + std::to_string(niter) + " iterations)");
    gft::ift::ORelax_1_Multi(A, fscn, pol * 100.0, S, label, niter);
    DebugTimer::getInstance().endEvent("ORelax_1_Multi (Relaxation - " + std::to_string(niter) + " iterations)");

    int max_label = gft::Scene32::GetMaximumValue(label);
    if (max_label <= 1)
    {
        DebugTimer::getInstance().startEvent("Dilation Conditional");
        dilation_conditional(fscn, label, 1 /*radius sphere adj */, percentile /*percentile*/);
        DebugTimer::getInstance().endEvent("Dilation Conditional");
    }
    else
    {
        std::cout << "Skipping binary dilation post-process for multi-label result (max label="
                  << max_label << ")." << std::endl;
    }

    gft::Scene32::Destroy(&fscn);

    end = clock();
    totaltime = ((double)(end - start)) / CLOCKS_PER_SEC;

    DebugTimer::getInstance().printSummary();

    gft::Scene32::Write(label, output_file);

    free(S);
    gft::Scene32::Destroy(&label);
    gft::AdjRel3::Destroy(&A);
    return 0;
}
