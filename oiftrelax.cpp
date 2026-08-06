
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

// Per-voxel morphological gradient G(p) = max_{q in A(p)} |I(p)-I(q)|. Used as the
// arc-cost image for the geodesic predecessor forest (SC_Pred_fsum), so geodesics run
// cheaply through flat bone and expensively across boundaries. High at object surfaces.
static gft::sScene32 *morph_gradient(gft::sScene32 *scn, gft::sAdjRel3 *A)
{
    gft::sScene32 *G = gft::Scene32::Create(scn);
    gft::Voxel u, v;
    for (int p = 0; p < scn->n; p++)
    {
        u.c.x = gft::Scene32::GetAddressX(scn, p);
        u.c.y = gft::Scene32::GetAddressY(scn, p);
        u.c.z = gft::Scene32::GetAddressZ(scn, p);
        int gmax = 0;
        for (int i = 1; i < A->n; i++)
        {
            v.v = u.v + A->d[i].v;
            if (gft::Scene32::IsValidVoxel(scn, v))
            {
                int q = gft::Scene32::GetVoxelAddress(scn, v);
                int d = abs(scn->data[p] - scn->data[q]);
                if (d > gmax) gmax = d;
            }
        }
        G->data[p] = gmax;
    }
    return G;
}

// 6-connected Dijkstra distance (mm) from the object-seed set. Approximate Euclidean
// distance transform; used by the Local Band thickness cap. Unreached voxels get INT_MAX.
static gft::sScene32 *seed_distance_mm(gft::sScene32 *tmpl, gft::sAdjRel3 *A, int *Sobj)
{
    int n = tmpl->n;
    float *cost = gft::AllocFloatArray(n);
    gft::sHeap *Q = gft::Heap::Create(n, cost);
    float *Dpq = (float *)malloc(A->n * sizeof(float));
    for (int i = 1; i < A->n; i++)
        Dpq[i] = sqrtf(A->d[i].axis.x * A->d[i].axis.x * tmpl->dx * tmpl->dx +
                       A->d[i].axis.y * A->d[i].axis.y * tmpl->dy * tmpl->dy +
                       A->d[i].axis.z * A->d[i].axis.z * tmpl->dz * tmpl->dz);
    for (int p = 0; p < n; p++)
        cost[p] = FLT_MAX;
    for (int i = 1; i <= Sobj[0]; i++)
    {
        cost[Sobj[i]] = 0.0f;
        gft::Heap::Insert_MinPolicy(Q, Sobj[i]);
    }
    gft::Voxel u, v;
    int p, q;
    while (!gft::Heap::IsEmpty(Q))
    {
        gft::Heap::Remove_MinPolicy(Q, &p);
        u.c.x = gft::Scene32::GetAddressX(tmpl, p);
        u.c.y = gft::Scene32::GetAddressY(tmpl, p);
        u.c.z = gft::Scene32::GetAddressZ(tmpl, p);
        for (int i = 1; i < A->n; i++)
        {
            v.v = u.v + A->d[i].v;
            if (gft::Scene32::IsValidVoxel(tmpl, v))
            {
                q = gft::Scene32::GetVoxelAddress(tmpl, v);
                if (Q->color[q] != BLACK)
                {
                    float tmp = cost[p] + Dpq[i];
                    if (tmp < cost[q])
                        gft::Heap::Update_MinPolicy(Q, q, tmp);
                }
            }
        }
    }
    gft::sScene32 *D = gft::Scene32::Create(tmpl);
    for (int p2 = 0; p2 < n; p2++)
        D->data[p2] = (cost[p2] >= FLT_MAX) ? INT_MAX : (int)(cost[p2] + 0.5f);
    free(Dpq);
    gft::FreeFloatArray(&cost);
    gft::Heap::Destroy(&Q);
    return D;
}

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
    int blur_passes = 2;      // Gaussian pre-smoothing passes (default 2 = historical double blur)
    bool use_gsc = false;     // --gsc: Geodesic Star Convexity shape gate (opt-in)
    float gsc_power = 1.0f;    // --gsc [power]: geodesic contrast exponent
    bool use_band = false;    // --band: Local Band object-thickness cap (opt-in)
    int band_dmax = 0;        // --band <dmax_mm>: max object distance (mm) from internal seeds
    char *dist_file = NULL;   // --dist-file: external distance field for the band gate (e.g. atlas-shaped)
    char *struct_file = NULL; // --struct-file: per-voxel bone structure-ID field for the cross-structure gate (opt-in)
    bool use_geo_tiebreak = false;  // --geo-tiebreak: geodesic plateau tie-break on object conquest (opt-in)
    int geo_tol = 0;                // --geo-tol N: near-equal arc tolerance for the geodesic tie-break (default 0 = exact)
    char *output_file;
    if (argc < 7)
    {
        fprintf(stdout, "usage:\n");
        fprintf(stdout, "oiftrelax <volume> <file_seeds> <pol> <niter> <percentile> <output_file> [boundary_stride] [pol_file] [--blur <n>]\n");
        fprintf(stdout, "\t pol.............. Global boundary polarity [-1.0, 1.0] (ignored if pol_file given)\n");
        fprintf(stdout, "\t niter............ Relaxation iterations (0 = unlimited)\n");
        fprintf(stdout, "\t percentile...... Dilation percentile (binary mode only)\n");
        fprintf(stdout, "\t output_file..... Output label file (e.g., label.nii.gz)\n");
        fprintf(stdout, "\t boundary_stride. Stride for auto boundary bg seeds (default=8, 0=off)\n");
        fprintf(stdout, "\t pol_file........ Per-class polarity file (optional). Format:\n");
        fprintf(stdout, "\t                  Line 1: <n_labels>\n");
        fprintf(stdout, "\t                  Line 2: pol_0 pol_1 pol_2 ... pol_{n-1}\n");
        fprintf(stdout, "\t                  Values in [-1.0, 1.0]. pol_0 = background.\n");
        fprintf(stdout, "\t --blur <n>....... Gaussian pre-smoothing passes (default=2). 1=light, 0=off.\n");
        fprintf(stdout, "\t                  Fewer passes preserve thin tubular structures (e.g.\n");
        fprintf(stdout, "\t                  distal airways) that the double blur otherwise erases.\n");
        fprintf(stdout, "\t --gsc [power].... Geodesic Star Convexity shape gate (opt-in, default off).\n");
        fprintf(stdout, "\t                  Constrains the object to be star-convex w.r.t. the internal\n");
        fprintf(stdout, "\t                  seeds; curbs leaks through gradient-free junctions. power\n");
        fprintf(stdout, "\t                  = geodesic contrast exponent (default 1.0).\n");
        fprintf(stdout, "\t --band <dmax>.... Local Band thickness cap (opt-in, default off). Object may\n");
        fprintf(stdout, "\t                  not grow beyond dmax mm from the internal seeds.\n");
        fprintf(stdout, "\t --dist-file <f>.. External distance field (NIfTI) for the band gate instead\n");
        fprintf(stdout, "\t                  of the seed distance; e.g. distance from a warped-atlas rib\n");
        fprintf(stdout, "\t                  shape, so --band walls growth to the atlas anatomy.\n");
        fprintf(stdout, "\t --struct-file <f> Per-voxel structure-ID field (NIfTI, 0 = none) for the\n");
        fprintf(stdout, "\t                  cross-structure gate (opt-in, default off). An object may not\n");
        fprintf(stdout, "\t                  conquer a voxel belonging to a different, non-zero structure;\n");
        fprintf(stdout, "\t                  growth into id 0 (soft tissue / marrow) stays allowed.\n");
        fprintf(stdout, "\t --geo-tiebreak... Geodesic plateau tie-break (opt-in, default off). On an\n");
        fprintf(stdout, "\t                  equal-weight object conquest the geodesically nearest seed\n");
        fprintf(stdout, "\t                  wins, so inter-object boundaries in uniform regions fall on\n");
        fprintf(stdout, "\t                  the watershed line instead of arbitrary queue order.\n");
        fprintf(stdout, "\t --geo-tol <N>.... Tolerance for --geo-tiebreak (default 0 = exact tie). Admits\n");
        fprintf(stdout, "\t                  near-equal arcs (cost within N) so a geodesically closer seed\n");
        fprintf(stdout, "\t                  can re-conquer; approximates a geodesic/watershed partition.\n");
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
    // Optional positional args 7 (boundary_stride) and 8 (pol_file). Skip them if
    // they look like a named flag ("--...") or are empty, so flags such as --blur
    // can follow the 6 required positional args directly.
    if (argc >= 8 && argv[7][0] != '-' && argv[7][0] != '\0')
        boundary_stride = atoi(argv[7]);

    // Per-class polarity file (optional 8th arg)
    char *pol_file = NULL;
    if (argc >= 9 && argv[8][0] != '-' && argv[8][0] != '\0')
        pol_file = argv[8];

    // Named flag: --blur <n> (number of Gaussian pre-smoothing passes).
    // Default 2 preserves the historical double-blur behavior; 1 = light, 0 = none.
    for (int ai = 7; ai < argc; ai++)
    {
        if (std::string(argv[ai]) == "--blur" && ai + 1 < argc)
        {
            blur_passes = atoi(argv[ai + 1]);
            if (blur_passes < 0)
                blur_passes = 0;
        }
        else if (std::string(argv[ai]) == "--gsc")
        {
            use_gsc = true;
            // Optional numeric power argument (default 1.0).
            if (ai + 1 < argc)
            {
                char c = argv[ai + 1][0];
                if ((c >= '0' && c <= '9') || c == '.')
                    gsc_power = atof(argv[ai + 1]);
            }
        }
        else if (std::string(argv[ai]) == "--band" && ai + 1 < argc)
        {
            use_band = true;
            band_dmax = atoi(argv[ai + 1]);
        }
        else if (std::string(argv[ai]) == "--dist-file" && ai + 1 < argc)
        {
            dist_file = argv[ai + 1];
        }
        else if (std::string(argv[ai]) == "--struct-file" && ai + 1 < argc)
        {
            struct_file = argv[ai + 1];
        }
        else if (std::string(argv[ai]) == "--geo-tiebreak")
        {
            use_geo_tiebreak = true;
        }
        else if (std::string(argv[ai]) == "--geo-tol" && ai + 1 < argc)
        {
            geo_tol = atoi(argv[ai + 1]);
        }
    }

    // Parse per-class polarity file
    std::vector<float> per_class_vec;
    bool use_per_class = false;
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
                    per_class_vec[ci] = pv * 100.0f;  // convert [-1,1] to [-100,100]
                }
                if (ok)
                {
                    use_per_class = true;
                    std::cout << "Per-class polarity loaded: " << n_labels << " classes [";
                    for (int ci = 0; ci < n_labels; ci++)
                    {
                        if (ci > 0) std::cout << ", ";
                        std::cout << std::fixed << std::setprecision(2) << per_class_vec[ci] / 100.0f;
                    }
                    std::cout << "]" << std::endl;
                }
            }
            fclose(pfp);
        }
        else
        {
            std::cerr << "Warning: could not open polarity file: " << pol_file << std::endl;
        }
    }

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

    // Optional Gaussian pre-smoothing. Each pass smooths the previous result.
    // blur_passes == 0 operates on the raw (intensity-corrected) volume, which
    // preserves thin tubular structures the double blur would otherwise wash out
    // before the OIFT boundary competition runs. The segmentation works on `fscn`;
    // the read volume `scn` is freed here (or reused directly when no blur).
    if (blur_passes <= 0)
    {
        std::cout << "Gaussian pre-smoothing: disabled (--blur 0)" << std::endl;
        fscn = scn;  // operate on the read volume directly; freed later via fscn
    }
    else
    {
        gft::sScene32 *cur = scn;
        for (int b = 0; b < blur_passes; b++)
        {
            std::string ev = "GaussianBlur (" + std::to_string(b + 1) + "x)";
            DebugTimer::getInstance().startEvent(ev);
            fscn = gft::Scene32::GaussianBlur(cur);
            DebugTimer::getInstance().endEvent(ev);
            if (cur != scn)
                gft::Scene32::Destroy(&cur);  // free intermediate blur result
            cur = fscn;
        }
        gft::Scene32::Destroy(&scn);  // free the original read volume
        fscn = cur;
    }

    DebugTimer::getInstance().startEvent("OIFT_Multi (Oriented Image Foresting Transform)");
    if (use_gsc || use_band || use_geo_tiebreak || struct_file != NULL)
    {
        // Build the object-only (label>0) seed set that roots the shape constraints.
        int *Sobj = (int *)calloc(S[0] + 1, sizeof(int));
        int nobj = 0;
        for (int si = 1; si <= S[0]; si++)
            if (label->data[S[si]] > 0)
                Sobj[++nobj] = S[si];
        Sobj[0] = nobj;

        gft::sScene32 *pred = NULL;
        gft::sScene32 *sdist = NULL;
        gft::sScene32 *structId = NULL;
        if (use_gsc)
        {
            if (use_per_class)
                std::cout << "Note: --gsc uses global polarity (" << pol
                          << "); per-class polarity ignored for the shape gate." << std::endl;
            gft::sScene32 *G = morph_gradient(fscn, A);
            pred = gft::ift::SC_Pred_fsum(G, A, Sobj, gsc_power);
            gft::Scene32::Destroy(&G);
            std::cout << "GSC enabled: geodesic star-convexity gate (power=" << gsc_power
                      << ", " << nobj << " object seeds)." << std::endl;
        }
        if (use_band)
        {
            if (dist_file != NULL)
            {
                // External distance field (e.g. distance from a warped-atlas rib shape):
                // the band walls object growth to within band_dmax mm of that shape,
                // rather than of the sparse seed cores. Must match the volume dimensions.
                sdist = gft::Scene32::Read(dist_file);
                if (sdist == NULL || sdist->n != label->n)
                {
                    std::cerr << "Error: --dist-file dimensions do not match volume (or unreadable): "
                              << dist_file << std::endl;
                    exit(1);
                }
                std::cout << "Local Band (external dist-file): cap dmax=" << band_dmax
                          << " mm from " << dist_file << std::endl;
            }
            else
            {
                sdist = seed_distance_mm(fscn, A, Sobj);
                std::cout << "Local Band enabled: object thickness cap dmax=" << band_dmax << " mm." << std::endl;
            }
        }
        if (struct_file != NULL)
        {
            // Per-voxel bone structure IDs (0 = none): the cross-structure gate forbids an
            // object from conquering a voxel that belongs to a different, non-zero structure.
            structId = gft::Scene32::Read(struct_file);
            if (structId == NULL || structId->n != label->n)
            {
                std::cerr << "Error: --struct-file dimensions do not match volume (or unreadable): "
                          << struct_file << std::endl;
                exit(1);
            }
            std::cout << "Cross-structure gate enabled: " << struct_file << std::endl;
        }
        if (use_geo_tiebreak)
            std::cout << "Geodesic tie-break enabled: equal-weight object plateaus resolved by nearest seed (tol=" << geo_tol << ")." << std::endl;
        gft::ift::OIFT_Multi_Constrained(A, fscn, pol * 100.0, S, label, pred, sdist, structId, band_dmax, use_geo_tiebreak ? 1 : 0, geo_tol);
        if (pred != NULL) gft::Scene32::Destroy(&pred);
        if (sdist != NULL) gft::Scene32::Destroy(&sdist);
        if (structId != NULL) gft::Scene32::Destroy(&structId);
        free(Sobj);
    }
    else if (use_per_class)
    {
        int ml = (int)per_class_vec.size() - 1;
        gft::ift::OIFT_Multi_PerClass(A, fscn, per_class_vec.data(), ml, S, label);
    }
    else
    {
        gft::ift::OIFT_Multi(A, fscn, pol * 100.0, S, label);
    }
    DebugTimer::getInstance().endEvent("OIFT_Multi (Oriented Image Foresting Transform)");

    DebugTimer::getInstance().startEvent("ORelax_1_Multi (Relaxation - " + std::to_string(niter) + " iterations)");
    if (use_per_class && !(use_gsc || use_band))
    {
        int ml = (int)per_class_vec.size() - 1;
        gft::ift::ORelax_1_Multi_PerClass(A, fscn, per_class_vec.data(), ml, S, label, niter);
    }
    else
    {
        gft::ift::ORelax_1_Multi(A, fscn, pol * 100.0, S, label, niter);
    }
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
