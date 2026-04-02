// exp_gaussian_rbf_relax.cpp
// Experiment 2: Gaussian RBF relaxation weights
// Replaces 8th-power weighting in ORelax with:
//   w = exp(-|I(p)-I(q)|^2 / sigma^2) / Dpq

#include "gft.h"

#include <algorithm>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <cmath>

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

// Estimate sigma from global intensity std dev
static float estimate_sigma(gft::sScene32 *scn)
{
    long long sum = 0, sum2 = 0;
    int n = scn->n;
    for (int p = 0; p < n; p++)
    {
        long long v = scn->data[p];
        sum += v;
        sum2 += v * v;
    }
    double mean = (double)sum / n;
    double var = (double)sum2 / n - mean * mean;
    if (var < 1.0) var = 1.0;
    return (float)sqrt(var);
}

// ==================== CUSTOM ORELAX WITH GAUSSIAN RBF ====================
static void ORelax_1_Multi_GaussRBF(gft::sAdjRel3 *A,
                                     gft::sScene32 *scn,
                                     float per,
                                     float sigma,
                                     int *S,
                                     gft::sScene32 *label,
                                     int ntimes)
{
    gft::sScene32 *mask;
    float *flabel_1, *flabel_2, *tmp;
    int *mask_nodes;
    int n, p, q, i, j, k, nlast, ninic;
    gft::Voxel u, v;
    float sw, w, per_pq, dmin;
    float *Dpq;
    std::vector<int> class_labels;
    std::vector<int> label_to_class;
    std::vector<char> present;
    int max_seed_label = 0;
    int nclasses;
    int bg_class = 0;
    size_t total;

    if (S == NULL || S[0] <= 0) return;

    float sigma2 = sigma * sigma;

    for (i = 1; i <= S[0]; i++)
    {
        p = S[i];
        if (p >= 0 && p < label->n)
            max_seed_label = MAX(max_seed_label, label->data[p]);
    }
    max_seed_label = MAX(0, max_seed_label);

    present.assign(max_seed_label + 1, 0);
    present[0] = 1;
    for (i = 1; i <= S[0]; i++)
    {
        p = S[i];
        if (p >= 0 && p < label->n)
        {
            int lb = label->data[p];
            if (lb >= 0 && lb <= max_seed_label)
                present[lb] = 1;
        }
    }

    for (i = 0; i <= max_seed_label; i++)
        if (present[i])
            class_labels.push_back(i);

    if (class_labels.empty()) class_labels.push_back(0);
    nclasses = (int)class_labels.size();
    label_to_class.assign(max_seed_label + 1, -1);
    for (i = 0; i < nclasses; i++)
        label_to_class[class_labels[i]] = i;
    bg_class = label_to_class[0];
    if (bg_class < 0) bg_class = 0;

    dmin = MIN(scn->dx, MIN(scn->dy, scn->dz));
    Dpq = (float *)malloc(A->n * sizeof(float));
    for (i = 1; i < A->n; i++)
    {
        Dpq[i] = sqrtf(A->d[i].axis.x * A->d[i].axis.x * scn->dx * scn->dx +
                       A->d[i].axis.y * A->d[i].axis.y * scn->dy * scn->dy +
                       A->d[i].axis.z * A->d[i].axis.z * scn->dz * scn->dz) / dmin;
    }

    ninic = 1;
    n = label->n;
    total = (size_t)n * (size_t)nclasses;
    flabel_1 = (float *)calloc(total, sizeof(float));
    flabel_2 = (float *)calloc(total, sizeof(float));
    mask_nodes = (int *)malloc(sizeof(int) * (n + 1));
    mask_nodes[0] = 0;
    mask = gft::Scene32::Create(label);
    gft::Scene32::Fill(mask, 0);

    for (p = 0; p < n; p++)
    {
        size_t poff = (size_t)p * (size_t)nclasses;
        int lb = label->data[p];
        int c = bg_class;
        if (lb >= 0 && lb <= max_seed_label && label_to_class[lb] >= 0)
            c = label_to_class[lb];
        flabel_1[poff + (size_t)c] = 1.0f;

        u.c.x = gft::Scene32::GetAddressX(label, p);
        u.c.y = gft::Scene32::GetAddressY(label, p);
        u.c.z = gft::Scene32::GetAddressZ(label, p);
        for (i = 1; i < A->n; i++)
        {
            v.v = u.v + A->d[i].v;
            if (gft::Scene32::IsValidVoxel(label, v))
            {
                q = gft::Scene32::GetVoxelAddress(label, v);
                if (label->data[p] != label->data[q])
                {
                    mask->data[p] = 1;
                    mask_nodes[0]++;
                    mask_nodes[mask_nodes[0]] = p;
                    break;
                }
            }
        }
    }

    while (ntimes > 0)
    {
        memcpy(flabel_2, flabel_1, total * sizeof(float));
        for (j = 1; j <= mask_nodes[0]; j++)
        {
            size_t poff;
            p = mask_nodes[j];
            u.c.x = gft::Scene32::GetAddressX(label, p);
            u.c.y = gft::Scene32::GetAddressY(label, p);
            u.c.z = gft::Scene32::GetAddressZ(label, p);
            poff = (size_t)p * (size_t)nclasses;
            for (k = 0; k < nclasses; k++)
                flabel_2[poff + (size_t)k] = 0.0f;
            sw = 0.0f;

            for (i = 1; i < A->n; i++)
            {
                v.v = u.v + A->d[i].v;
                if (gft::Scene32::IsValidVoxel(label, v))
                {
                    size_t qoff;
                    q = gft::Scene32::GetVoxelAddress(label, v);
                    qoff = (size_t)q * (size_t)nclasses;

                    float diff = (float)(scn->data[p] - scn->data[q]);
                    // Gaussian RBF: w = exp(-diff^2 / sigma^2) / Dpq
                    w = expf(-(diff * diff) / sigma2) / Dpq[i];

                    sw += w;
                    for (k = 0; k < nclasses; k++)
                        flabel_2[poff + (size_t)k] += w * flabel_1[qoff + (size_t)k];
                }
            }

            if (sw > 0.0f)
                for (k = 0; k < nclasses; k++)
                    flabel_2[poff + (size_t)k] /= sw;
            else
                for (k = 0; k < nclasses; k++)
                    flabel_2[poff + (size_t)k] = flabel_1[poff + (size_t)k];
        }

        tmp = flabel_1; flabel_1 = flabel_2; flabel_2 = tmp;

        for (i = 1; i <= S[0]; i++)
        {
            size_t poff;
            int lb, c = bg_class;
            p = S[i];
            if (p < 0 || p >= n) continue;
            lb = label->data[p];
            if (lb >= 0 && lb <= max_seed_label && label_to_class[lb] >= 0)
                c = label_to_class[lb];
            poff = (size_t)p * (size_t)nclasses;
            for (k = 0; k < nclasses; k++)
                flabel_1[poff + (size_t)k] = 0.0f;
            flabel_1[poff + (size_t)c] = 1.0f;
        }

        ntimes--;

        if (ntimes > 0)
        {
            nlast = mask_nodes[0];
            for (j = ninic; j <= mask_nodes[0]; j++)
            {
                p = mask_nodes[j];
                u.c.x = gft::Scene32::GetAddressX(label, p);
                u.c.y = gft::Scene32::GetAddressY(label, p);
                u.c.z = gft::Scene32::GetAddressZ(label, p);
                for (i = 1; i < A->n; i++)
                {
                    v.v = u.v + A->d[i].v;
                    if (gft::Scene32::IsValidVoxel(label, v))
                    {
                        q = gft::Scene32::GetVoxelAddress(label, v);
                        if (mask->data[q] == 0)
                        {
                            mask->data[q] = 1;
                            nlast++;
                            mask_nodes[nlast] = q;
                        }
                    }
                }
            }
            ninic = mask_nodes[0] + 1;
            mask_nodes[0] = nlast;
        }
    }

    for (p = 0; p < n; p++)
    {
        size_t poff = (size_t)p * (size_t)nclasses;
        int best = 0;
        float bestv = flabel_1[poff];
        for (k = 1; k < nclasses; k++)
        {
            float vprob = flabel_1[poff + (size_t)k];
            if (vprob > bestv) { bestv = vprob; best = k; }
        }
        label->data[p] = class_labels[best];
    }

    free(Dpq);
    free(flabel_1);
    free(flabel_2);
    free(mask_nodes);
    gft::Scene32::Destroy(&mask);
}

int main(int argc, char **argv)
{
    std::cout << "=== EXP 2: Gaussian RBF relaxation weights ===" << std::endl;

    gft::sScene32 *scn, *fscn, *label;
    gft::sAdjRel3 *A;
    FILE *fp;
    int *S;
    int p, i, j, nseeds, x, y, z, id, lb, Imin;
    int niter = 50;
    float pol = 0.5f;
    int percentile = 50;
    int boundary_stride = 8;
    float sigma = -1.0f; // auto
    char *output_file;

    if (argc < 7)
    {
        fprintf(stdout, "usage:\n");
        fprintf(stdout, "exp_gaussian_rbf_relax <volume> <file_seeds> <pol> <niter> <percentile> <output_file> [boundary_stride] [pol_file] [--sigma <val>]\n");
        exit(0);
    }

    pol = atof(argv[3]);
    niter = atoi(argv[4]);
    percentile = atoi(argv[5]);
    output_file = argv[6];
    if (argc >= 8) boundary_stride = atoi(argv[7]);

    // Parse --sigma from remaining args
    for (int ai = 9; ai < argc; ai++)
    {
        if (std::string(argv[ai]) == "--sigma" && ai + 1 < argc)
        {
            sigma = atof(argv[ai + 1]);
            ai++;
        }
    }

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

    // Auto-estimate sigma if not provided
    if (sigma <= 0.0f)
    {
        sigma = estimate_sigma(fscn);
        std::cout << "Auto sigma = " << sigma << std::endl;
    }
    else
    {
        std::cout << "Sigma (RBF): " << sigma << std::endl;
    }

    DebugTimer::getInstance().startEvent("OIFT_Multi (standard)");
    gft::ift::OIFT_Multi(A, fscn, pol * 100.0f, S, label);
    DebugTimer::getInstance().endEvent("OIFT_Multi (standard)");

    DebugTimer::getInstance().startEvent("ORelax_1_Multi_GaussRBF");
    ORelax_1_Multi_GaussRBF(A, fscn, pol * 100.0f, sigma, S, label, niter);
    DebugTimer::getInstance().endEvent("ORelax_1_Multi_GaussRBF");

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
