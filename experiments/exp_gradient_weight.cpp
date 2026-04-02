// exp_gradient_weight.cpp
// Experiment 1: Gradient-magnitude blended edge weights
// Blends intensity-difference with 3D gradient magnitude:
//   w = alpha*|I(p)-I(q)| + (1-alpha)*(grad_mag[p]+grad_mag[q])/2

#include "gft.h"
#include "gft_gradient3.h"

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

// ==================== CUSTOM OIFT WITH GRADIENT WEIGHT ====================
// Pre-normalized gradient magnitude scene (same dims as scn, values in [0, Wmax])
static void OIFT_Multi_GradWeight(gft::sAdjRel3 *A,
                                   gft::sScene32 *scn,
                                   gft::sScene32 *grad_norm,
                                   float per,
                                   float alpha,
                                   int *S,
                                   gft::sScene32 *label)
{
    gft::sPQueue32 *Q = NULL;
    int i, p, q, n;
    int w, Wmax;
    gft::sScene32 *value;
    gft::Voxel u, v;
    float per_pq;

    value = gft::Scene32::Create(scn);
    n = label->n;
    Wmax = gft::Scene32::GetMaximumValue(scn);
    Wmax = (int)(Wmax * (1.0f + fabsf(per) / 100.0f));
    // Wmax after blend can be at most Wmax (grad_norm already in [0,Wmax])
    Q = gft::PQueue32::Create(Wmax + 2, n, value->data);

    for (p = 0; p < n; p++)
        value->data[p] = (label->data[p] == NIL) ? INT_MAX : 0;

    if (S != NULL)
        for (i = 1; i <= S[0]; i++)
            gft::PQueue32::FastInsertElem(Q, S[i]);
    else
        for (p = 0; p < n; p++)
            if (label->data[p] != NIL)
                gft::PQueue32::FastInsertElem(Q, p);

    while (!gft::PQueue32::IsEmpty(Q))
    {
        p = gft::PQueue32::FastRemoveMinFIFO(Q);
        u.c.x = gft::Scene32::GetAddressX(label, p);
        u.c.y = gft::Scene32::GetAddressY(label, p);
        u.c.z = gft::Scene32::GetAddressZ(label, p);

        for (i = 1; i < A->n; i++)
        {
            v.v = u.v + A->d[i].v;
            if (gft::Scene32::IsValidVoxel(label, v))
            {
                q = gft::Scene32::GetVoxelAddress(label, v);
                if (Q->L.elem[q].color != BLACK)
                {
                    float intensity_diff = (float)abs(scn->data[p] - scn->data[q]);
                    float grad_avg = (grad_norm->data[p] + grad_norm->data[q]) * 0.5f;
                    float wf = alpha * intensity_diff + (1.0f - alpha) * grad_avg;
                    w = (int)wf;

                    per_pq = (label->data[p] == 0) ? -per : per;
                    if (scn->data[p] > scn->data[q])
                        w = (int)(wf * (1.0f + per_pq / 100.0f));
                    else if (scn->data[p] < scn->data[q])
                        w = (int)(wf * (1.0f - per_pq / 100.0f));
                    if (w < 0) w = 0;

                    if (w < value->data[q])
                    {
                        if (Q->L.elem[q].color == GRAY)
                            gft::PQueue32::FastRemoveElem(Q, q);
                        value->data[q] = w;
                        label->data[q] = label->data[p];
                        gft::PQueue32::FastInsertElem(Q, q);
                    }
                }
            }
        }
    }
    gft::Scene32::Destroy(&value);
    gft::PQueue32::Destroy(&Q);
}

static void ORelax_1_Multi_GradWeight(gft::sAdjRel3 *A,
                                       gft::sScene32 *scn,
                                       gft::sScene32 *grad_norm,
                                       float per,
                                       float alpha,
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
    float Wmax;
    float *Dpq;
    std::vector<int> class_labels;
    std::vector<int> label_to_class;
    std::vector<char> present;
    int max_seed_label = 0;
    int nclasses;
    int bg_class = 0;
    size_t total;

    if (S == NULL || S[0] <= 0) return;

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

    Wmax = (float)gft::Scene32::GetMaximumValue(scn);
    Wmax *= (1.0f + fabsf(per) / 100.0f);
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
                    float p_bg;
                    q = gft::Scene32::GetVoxelAddress(label, v);
                    qoff = (size_t)q * (size_t)nclasses;

                    float intensity_diff = fabsf((float)(scn->data[p] - scn->data[q]));
                    float grad_avg = (grad_norm->data[p] + grad_norm->data[q]) * 0.5f;
                    float wf = alpha * intensity_diff + (1.0f - alpha) * grad_avg;

                    p_bg = flabel_1[qoff + (size_t)bg_class];
                    per_pq = per * (2.0f * p_bg - 1.0f);
                    if (scn->data[p] > scn->data[q])
                        wf *= (1.0f + per_pq / 100.0f);
                    else if (scn->data[p] < scn->data[q])
                        wf *= (1.0f - per_pq / 100.0f);
                    if (wf < 0.0f) wf = 0.0f;

                    w = Wmax - wf;
                    if (w < 0.0f) w = 0.0f;
                    w = w * w; w = w * w; w = w * w;
                    w /= Dpq[i];

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
    std::cout << "=== EXP 1: Gradient-magnitude blended edge weights ===" << std::endl;

    gft::sScene32 *scn, *fscn, *label, *grad_norm;
    gft::sAdjRel3 *A;
    FILE *fp;
    int *S;
    int p, i, j, nseeds, x, y, z, id, lb, Imin;
    int niter = 50;
    float pol = 0.5f;
    int percentile = 50;
    int boundary_stride = 8;
    float alpha = 0.5f;
    char *output_file;

    if (argc < 7)
    {
        fprintf(stdout, "usage:\n");
        fprintf(stdout, "exp_gradient_weight <volume> <file_seeds> <pol> <niter> <percentile> <output_file> [boundary_stride] [pol_file] [--alpha <val>]\n");
        exit(0);
    }

    pol = atof(argv[3]);
    niter = atoi(argv[4]);
    percentile = atoi(argv[5]);
    output_file = argv[6];
    if (argc >= 8) boundary_stride = atoi(argv[7]);

    // Parse --alpha from remaining args
    for (int ai = 9; ai < argc; ai++)
    {
        if (std::string(argv[ai]) == "--alpha" && ai + 1 < argc)
        {
            alpha = atof(argv[ai + 1]);
            ai++;
        }
    }
    std::cout << "Alpha (blend): " << alpha << std::endl;

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

    // Compute gradient magnitude and normalize to [0, Wmax]
    DebugTimer::getInstance().startEvent("Sobel gradient + normalize");
    grad_norm = gft::Scene32::SobelFilter(fscn);
    {
        int Gmax = gft::Scene32::GetMaximumValue(grad_norm);
        int Wmax = gft::Scene32::GetMaximumValue(fscn);
        if (Gmax > 0 && Wmax > 0)
        {
            for (p = 0; p < grad_norm->n; p++)
            {
                long long val = (long long)grad_norm->data[p] * Wmax / Gmax;
                grad_norm->data[p] = (int)val;
            }
        }
    }
    DebugTimer::getInstance().endEvent("Sobel gradient + normalize");

    DebugTimer::getInstance().startEvent("OIFT_Multi_GradWeight");
    OIFT_Multi_GradWeight(A, fscn, grad_norm, pol * 100.0f, alpha, S, label);
    DebugTimer::getInstance().endEvent("OIFT_Multi_GradWeight");

    DebugTimer::getInstance().startEvent("ORelax_1_Multi_GradWeight");
    ORelax_1_Multi_GradWeight(A, fscn, grad_norm, pol * 100.0f, alpha, S, label, niter);
    DebugTimer::getInstance().endEvent("ORelax_1_Multi_GradWeight");

    gft::Scene32::Destroy(&grad_norm);

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
