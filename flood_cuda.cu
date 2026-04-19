/*
 * NOTE: READ CAREFULLY
 * Here the function `do_compute` is just a copy of the CPU sequential version.
 * Implement your GPU code with CUDA here. Check the README for further instructions.
 * You can modify everything in this file, as long as we can compile the executable using
 * this source code, and Makefile.
 *
 * Simulation of rainwater flooding
 * CUDA version
 *
 * Adapted for ACCE at the VU, Period 5 2025-2026 from the original version by
 * Based on the EduHPC 2025: Peachy assignment, Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2024/2025
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Headers for the CUDA assignment versions */
#include <cuda.h>

/* Example of macros for error checking in CUDA */
#define CUDA_CHECK_FUNCTION(call)                                                                                      \
    {                                                                                                                  \
        cudaError_t check = call;                                                                                      \
        if (check != cudaSuccess)                                                                                      \
            fprintf(stderr, "CUDA Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check));                      \
    }
#define CUDA_CHECK_KERNEL()                                                                                            \
    {                                                                                                                  \
        cudaError_t check = cudaGetLastError();                                                                        \
        if (check != cudaSuccess)                                                                                      \
            fprintf(stderr, "CUDA Kernel Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check));               \
    }

/*
 * Utils: Random generator
 */
#include "rng.c"

/*
 * Header file: Contains constants and definitions
 */
#include "flood.h"

extern "C" double get_time();

/* ───────────────────────── Types ───────────────────────── */

typedef struct {
    float x0, y0;
    float dx, dy;
    float radius;
    float intensity;
} CloudPrecomp_t;

/* ────────────────── Device helper: float atomicMax ────────────────── */

__device__ float atomicMaxFloat(float *addr, float value) {
    int *addr_as_int = (int *)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(__int_as_float(assumed), value)));
    } while (assumed != old);
    return __int_as_float(old);
}

/* ────────────────── Kernel 1: Rainfall ────────────────── */

__device__ float rainFromCloud(int minute, int row, int col,
                               const CloudPrecomp_t *cloud, float ex_factor,
                               int rows, int columns) {
    float cloud_x = cloud->x0 + (minute + 1) * cloud->dx;
    float cloud_y = cloud->y0 + (minute + 1) * cloud->dy;

    float col_start = COORD_SCEN2MAT_X(MAX(0, cloud_x - cloud->radius));
    float row_start = COORD_SCEN2MAT_Y(MAX(0, cloud_y - cloud->radius));
    float col_end = COORD_SCEN2MAT_X(MIN(cloud_x + cloud->radius, SCENARIO_SIZE));
    float row_end = COORD_SCEN2MAT_Y(MIN(cloud_y + cloud->radius, SCENARIO_SIZE));

    float frac_row = row_start - (int)row_start;
    float frac_col = col_start - (int)col_start;
    float row_pos = (float)row + frac_row;
    float col_pos = (float)col + frac_col;

    if (row_pos < row_start || row_pos >= row_end ||
        col_pos < col_start || col_pos >= col_end)
        return 0.0f;

    float x_pos = COORD_MAT2SCEN_X(col_pos);
    float y_pos = COORD_MAT2SCEN_Y(row_pos);
    float dist = sqrtf((x_pos - cloud_x) * (x_pos - cloud_x) +
                       (y_pos - cloud_y) * (y_pos - cloud_y));

    if (dist >= cloud->radius)
        return 0.0f;

    float rain = ex_factor *
                 MAX(0, cloud->intensity - dist / cloud->radius * sqrtf(cloud->intensity));
    return rain / 1000.0f / 60.0f;
}

__global__ void rainfall_kernel(int minute, int *water_level,
                                const CloudPrecomp_t *clouds, int num_clouds,
                                float ex_factor, int rows, int columns,
                                long *d_total_rain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * columns) return;

    int row = idx / columns;
    int col = idx % columns;

    int total = 0;
    for (int c = 0; c < num_clouds; c++) {
        float rain = rainFromCloud(minute, row, col, &clouds[c],
                                   ex_factor, rows, columns);
        if (rain > 0.0f)
            total += FIXED(rain);
    }
    if (total > 0) {
        water_level[idx] += total;
        atomicAdd((unsigned long long *)d_total_rain, (unsigned long long)total);
    }
}

/* ────────────────── Kernel 2: Spillage computation ────────────────── */

__global__ void spillage_kernel(const int *water_level, const float *ground,
                                float *spillage_flag, float *spillage_level,
                                float *spillage_from_neigh,
                                int rows, int columns,
                                long *d_total_water_loss) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * columns) return;

    int row = idx / columns;
    int col = idx % columns;

    if (water_level[idx] <= 0) return;

    float current_height = ground[idx] + FLOATING(water_level[idx]);
    float sum_diff = 0.0f;
    float my_spillage_level = 0.0f;

    for (int d = 0; d < CONTIGUOUS_CELLS; d++) {
        int nr = row + displacements[d][0];
        int nc = col + displacements[d][1];

        float neighbor_height;
        if (nr < 0 || nr >= rows || nc < 0 || nc >= columns)
            neighbor_height = ground[idx];
        else
            neighbor_height = ground[nr * columns + nc] + FLOATING(water_level[nr * columns + nc]);

        if (current_height >= neighbor_height) {
            float height_diff = current_height - neighbor_height;
            sum_diff += height_diff;
            my_spillage_level = fmaxf(my_spillage_level, height_diff);
        }
    }

    my_spillage_level = fminf(FLOATING(water_level[idx]), my_spillage_level);

    if (sum_diff <= 0.0f) return;

    float proportion = my_spillage_level / sum_diff;
    if (proportion <= 1e-8f) return;

    spillage_flag[idx] = 1.0f;
    spillage_level[idx] = my_spillage_level;

    int depths = CONTIGUOUS_CELLS;
    for (int d = 0; d < CONTIGUOUS_CELLS; d++) {
        int nr = row + displacements[d][0];
        int nc = col + displacements[d][1];

        float neighbor_height;
        if (nr < 0 || nr >= rows || nc < 0 || nc >= columns) {
            neighbor_height = ground[idx];
            if (current_height >= neighbor_height) {
                long loss = FIXED(proportion * (current_height - neighbor_height) / 2);
                atomicAdd((unsigned long long *)d_total_water_loss, (unsigned long long)loss);
            }
        } else {
            neighbor_height = ground[nr * columns + nc] + FLOATING(water_level[nr * columns + nc]);
            if (current_height >= neighbor_height) {
                int neigh_idx = nr * columns * depths + nc * depths + d;
                spillage_from_neigh[neigh_idx] = proportion * (current_height - neighbor_height);
            }
        }
    }
}

/* ────────────────── Kernel 3: Propagation ────────────────── */

__global__ void propagation_kernel(int *water_level,
                                   const float *spillage_flag,
                                   const float *spillage_level,
                                   const float *spillage_from_neigh,
                                   int rows, int columns,
                                   float *d_max_spillage) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * columns) return;

    if (spillage_flag[idx] == 1.0f) {
        float sl = spillage_level[idx];
        water_level[idx] -= FIXED(sl / SPILLAGE_FACTOR);
        atomicMaxFloat(d_max_spillage, sl / SPILLAGE_FACTOR);
    }

    int depths = 4;
    int base = idx * depths;
    for (int d = 0; d < 4; d++) {
        water_level[idx] += FIXED(spillage_from_neigh[base + d] / SPILLAGE_FACTOR);
    }
}

/* ────────────────── Main compute function ────────────────── */

extern "C" void do_compute(struct parameters *p, struct results *r) {
    int rows = p->rows, columns = p->columns;
    int *minute = &r->minute;

    CUDA_CHECK_FUNCTION(cudaSetDevice(0));
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

    size_t grid_size = (size_t)rows * (size_t)columns;

    /* Precompute cloud movement deltas on host */
    CloudPrecomp_t *clouds_pre = (CloudPrecomp_t *)malloc(sizeof(CloudPrecomp_t) * p->num_clouds);
    for (int c = 0; c < p->num_clouds; c++) {
        Cloud_t *orig = &p->clouds[c];
        clouds_pre[c].x0 = orig->x;
        clouds_pre[c].y0 = orig->y;
        clouds_pre[c].dx = (orig->speed / 60.0f) * cosf(orig->angle * (float)M_PI / 180.0f);
        clouds_pre[c].dy = (orig->speed / 60.0f) * sinf(orig->angle * (float)M_PI / 180.0f);
        clouds_pre[c].radius = orig->radius;
        clouds_pre[c].intensity = orig->intensity;
    }

    /* Device memory: grid arrays */
    int *d_water_level;
    float *d_ground;
    float *d_spillage_flag;
    float *d_spillage_level;
    float *d_spillage_from_neigh;

    CUDA_CHECK_FUNCTION(cudaMalloc(&d_water_level, sizeof(int) * grid_size));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_ground, sizeof(float) * grid_size));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_spillage_flag, sizeof(float) * grid_size));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_spillage_level, sizeof(float) * grid_size));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_spillage_from_neigh, sizeof(float) * grid_size * 4));

    CUDA_CHECK_FUNCTION(cudaMemset(d_water_level, 0, sizeof(int) * grid_size));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_ground, p->ground, sizeof(float) * grid_size, cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemset(d_spillage_flag, 0, sizeof(float) * grid_size));
    CUDA_CHECK_FUNCTION(cudaMemset(d_spillage_level, 0, sizeof(float) * grid_size));
    CUDA_CHECK_FUNCTION(cudaMemset(d_spillage_from_neigh, 0, sizeof(float) * grid_size * 4));

    /* Device memory: clouds */
    CloudPrecomp_t *d_clouds_pre;
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_clouds_pre, sizeof(CloudPrecomp_t) * p->num_clouds));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_clouds_pre, clouds_pre,
                                   sizeof(CloudPrecomp_t) * p->num_clouds, cudaMemcpyHostToDevice));

    /* Device memory: scalar accumulators */
    long *d_total_rain;
    long *d_total_water_loss;
    float *d_max_spillage;

    CUDA_CHECK_FUNCTION(cudaMalloc(&d_total_rain, sizeof(long)));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_total_water_loss, sizeof(long)));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_max_spillage, sizeof(float)));

    long zero_long = 0;
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_total_rain, &zero_long, sizeof(long), cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_total_water_loss, &zero_long, sizeof(long), cudaMemcpyHostToDevice));

    /* Host buffer for DEBUG prints and final stats */
    int *water_level = (int *)malloc(sizeof(int) * grid_size);

    int threads_per_block = 256;
    int num_blocks = ((int)grid_size + threads_per_block - 1) / threads_per_block;

#ifdef DEBUG
    print_matrix(PRECISION_FLOAT, rows, columns, p->ground, "Ground heights");
#ifndef ANIMATION
    print_clouds(p->num_clouds, p->clouds);
#endif
#endif

    double max_spillage_iter = DBL_MAX;

    /* Prepare to measure runtime */
    r->runtime = get_time();

    /* Flood simulation */
    for (*minute = 0; *minute < p->num_minutes && max_spillage_iter > p->threshold; (*minute)++) {

        /* Step 1 (GPU): Rainfall */
        rainfall_kernel<<<num_blocks, threads_per_block>>>(
            *minute, d_water_level, d_clouds_pre, p->num_clouds,
            p->ex_factor, rows, columns, d_total_rain);
        CUDA_CHECK_KERNEL();

#ifdef DEBUG
        CUDA_CHECK_FUNCTION(cudaMemcpy(water_level, d_water_level,
                                       sizeof(int) * grid_size, cudaMemcpyDeviceToHost));
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after rain");
#endif

        /* Step 2 (GPU): Spillage computation */
        spillage_kernel<<<num_blocks, threads_per_block>>>(
            d_water_level, d_ground,
            d_spillage_flag, d_spillage_level, d_spillage_from_neigh,
            rows, columns, d_total_water_loss);
        CUDA_CHECK_KERNEL();

        /* Step 3 (GPU): Propagation */
        float zero_f = 0.0f;
        CUDA_CHECK_FUNCTION(cudaMemcpy(d_max_spillage, &zero_f, sizeof(float), cudaMemcpyHostToDevice));

        propagation_kernel<<<num_blocks, threads_per_block>>>(
            d_water_level,
            d_spillage_flag, d_spillage_level, d_spillage_from_neigh,
            rows, columns, d_max_spillage);
        CUDA_CHECK_KERNEL();

        /* Read back max_spillage for termination check and statistics */
        float max_spillage_f;
        CUDA_CHECK_FUNCTION(cudaMemcpy(&max_spillage_f, d_max_spillage, sizeof(float), cudaMemcpyDeviceToHost));
        max_spillage_iter = (double)max_spillage_f;

        if (max_spillage_f > r->max_spillage_scenario) {
            r->max_spillage_scenario = max_spillage_f;
            r->max_spillage_minute = *minute;
        }

#ifdef DEBUG
#ifndef ANIMATION
        CUDA_CHECK_FUNCTION(cudaMemcpy(water_level, d_water_level,
                                       sizeof(int) * grid_size, cudaMemcpyDeviceToHost));
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after spillage");
#endif
#endif

        /* Reset spillage structures */
        CUDA_CHECK_FUNCTION(cudaMemset(d_spillage_flag, 0, sizeof(float) * grid_size));
        CUDA_CHECK_FUNCTION(cudaMemset(d_spillage_level, 0, sizeof(float) * grid_size));
        CUDA_CHECK_FUNCTION(cudaMemset(d_spillage_from_neigh, 0, sizeof(float) * grid_size * 4));
    }

    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
    r->runtime = get_time() - r->runtime;

    /* Copy final water_level back to host for stats and optional output */
    CUDA_CHECK_FUNCTION(cudaMemcpy(water_level, d_water_level,
                                   sizeof(int) * grid_size, cudaMemcpyDeviceToHost));

    /* Copy scalar accumulators back */
    CUDA_CHECK_FUNCTION(cudaMemcpy(&r->total_rain, d_total_rain, sizeof(long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&r->total_water_loss, d_total_water_loss, sizeof(long), cudaMemcpyDeviceToHost));

    if (p->final_matrix) {
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after spillage");
    }

    /* Statistics: Total remaining water and maximum amount of water in a cell */
    r->max_water_scenario = 0.0;
    for (int row_pos = 0; row_pos < rows; row_pos++) {
        for (int col_pos = 0; col_pos < columns; col_pos++) {
            int wl = water_level[row_pos * columns + col_pos];
            if (FLOATING(wl) > r->max_water_scenario)
                r->max_water_scenario = FLOATING(wl);
            r->total_water += wl;
        }
    }

    /* Free device resources */
    CUDA_CHECK_FUNCTION(cudaFree(d_water_level));
    CUDA_CHECK_FUNCTION(cudaFree(d_ground));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_flag));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_level));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_from_neigh));
    CUDA_CHECK_FUNCTION(cudaFree(d_clouds_pre));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_rain));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_water_loss));
    CUDA_CHECK_FUNCTION(cudaFree(d_max_spillage));

    /* Free host resources */
    free(clouds_pre);
    free(water_level);
}
