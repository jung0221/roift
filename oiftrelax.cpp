
#include "gft.h"

#include <algorithm>
#include <vector>
#include <iostream>

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
    char *output_file;
    if (argc < 7)
    {
        fprintf(stdout, "usage:\n");
        fprintf(stdout, "oiftrelax <volume> <file_seeds> <pol> <niter> <percentile>\n");
        fprintf(stdout, "\t pol.... Boundary polarity. It can be in the range [-1.0, 1.0]\n");
        fprintf(stdout, "\t niter.. Number of iterations of the relaxation procedure.\n");
        fprintf(stdout, "\t output_file.. Output label file name (e.g., label.nii.gz)\n");
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

    std::cout << argv[6] << std::endl;

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
    for (i = 0; i < nseeds; i++)
    {
        fscanf(fp, " %d %d %d %d %d", &x, &y, &z, &id, &lb);
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

    start = clock();

    fscn = gft::Scene32::GaussianBlur(scn);
    scn = fscn;
    fscn = gft::Scene32::GaussianBlur(scn);
    //-----------------
    // - radius sphere adjacency
    gft::Scene32::Destroy(&scn);
    gft::ift::OIFT(A, fscn, pol * 100.0, S, label);
    //-----------------
    gft::ift::ORelax_1(A, fscn, pol * 100.0, S, label, niter);
    // - percentile, % of dark intensities in dilation to be added as object
    dilation_conditional(fscn, label, 1 /*radius sphere adj */, percentile /*percentile*/);
    gft::Scene32::Destroy(&fscn);

    end = clock();
    totaltime = ((double)(end - start)) / CLOCKS_PER_SEC;
    // printf("Time: %f sec\n", totaltime);

    gft::Scene32::Write(label, output_file);

    free(S);
    gft::Scene32::Destroy(&scn);
    gft::Scene32::Destroy(&label);
    gft::AdjRel3::Destroy(&A);
    return 0;
}
