/*
 * NOTE: READ CAREFULLY
 * Here the function `do_compute` is just a copy of the CPU sequential version.
 * Implement your GPU code with CUDA here. Check the README for further instructions.
 * You can modify everything in this file, as long as we can compile the executable using
 * this source code, and Makefile.
 *
 * Simulation of rainwater flooding
 * CUDA version (Implement your parallel version here)
 *
 * Adapted for ACCE at the VU, Period 5 2025-2026 from the original version by
 * Based on the EduHPC 2025: Peachy assignment, Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2024/2025
 */

#include <float.h>
#include <string.h>
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

__global__ void step3_propagation_kernel(int rows,
                                         int columns,
                                         int *water_level,
                                         float *spillage_flag,
                                         float *spillage_level,
                                         float *spillage_from_neigh,
                                         int *max_spillage_bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = rows * columns;
    if (idx >= total_cells)
        return;

    if (spillage_flag[idx] == 1.0f) {
        float current_spillage = spillage_level[idx] / SPILLAGE_FACTOR;
        water_level[idx] -= FIXED(current_spillage);
        atomicMax(max_spillage_bits, __float_as_int(current_spillage));
    }

    int base = idx * CONTIGUOUS_CELLS;
    for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
        water_level[idx] += FIXED(spillage_from_neigh[base + cell_pos] / SPILLAGE_FACTOR);
    }

    /* Opt 1: Reset after read, eliminates separate reset kernel launch per iteration */
    spillage_flag[idx] = 0.0f;
    spillage_level[idx] = 0.0f;
    for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++)
        spillage_from_neigh[base + cell_pos] = 0.0f;
}


__global__ void rainfall_kernel(int rows,
                                int columns,
                                float row_start,
                                int row_count,
                                float col_start,
                                int col_count,
                                Cloud_t cloud,
                                float ex_factor,
                                int *water_level,
                                unsigned long long *total_rain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = row_count * col_count;
    if (idx >= total_cells)
        return;

    float row_pos = row_start + (float)(idx / col_count);
    float col_pos = col_start + (float)(idx % col_count);
    int row = (int)row_pos;
    int col = (int)col_pos;

    float x_pos = COORD_MAT2SCEN_X(col_pos);
    float y_pos = COORD_MAT2SCEN_Y(row_pos);
    float dx = x_pos - cloud.x;
    float dy = y_pos - cloud.y;
    float distance = sqrtf(dx * dx + dy * dy);

    if (distance < cloud.radius) {
        float rain = ex_factor * MAX(0.0f, cloud.intensity - distance / cloud.radius * sqrtf(cloud.intensity));
        float meters_per_minute = rain / 1000.0f / 60.0f;
        int rain_fixed = FIXED(meters_per_minute);
        atomicAdd(&water_level[row * columns + col], rain_fixed);
        atomicAdd(total_rain, (unsigned long long)rain_fixed);
    }
}

__global__ void step2_spillage_kernel(int rows,
                                      int columns,
                                      const float *ground,
                                      const int *water_level,
                                      float *spillage_flag,
                                      float *spillage_level,
                                      float *spillage_from_neigh,
                                      unsigned long long *total_water_loss) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = rows * columns;
    if (idx >= total_cells)
        return;

    if (water_level[idx] <= 0)
        return;

    int row = idx / columns;
    int col = idx % columns;

    float sum_diff = 0.0f;
    float my_spillage_level = 0.0f;
    float current_height = ground[idx] + FLOATING(water_level[idx]);

    for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
        int new_row = row + displacements[cell_pos][0];
        int new_col = col + displacements[cell_pos][1];

        float neighbor_height;
        if (new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns) {
            neighbor_height = ground[idx];
        } else {
            int neigh_idx = new_row * columns + new_col;
            neighbor_height = ground[neigh_idx] + FLOATING(water_level[neigh_idx]);
        }

        if (current_height >= neighbor_height) {
            float height_diff = current_height - neighbor_height;
            sum_diff += height_diff;
            my_spillage_level = MAX(my_spillage_level, height_diff);
        }
    }

    my_spillage_level = MIN(FLOATING(water_level[idx]), my_spillage_level);

    if (sum_diff > 0.0f) {
        float proportion = my_spillage_level / sum_diff;
        if (proportion > 1e-8f) {
            spillage_flag[idx] = 1.0f;
            spillage_level[idx] = my_spillage_level;

            for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
                int new_row = row + displacements[cell_pos][0];
                int new_col = col + displacements[cell_pos][1];

                float neighbor_height;
                if (new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns) {
                    neighbor_height = ground[idx];
                    if (current_height >= neighbor_height) {
                        unsigned long long loss =
                            (unsigned long long)FIXED(proportion * (current_height - neighbor_height) / 2.0f);
                        atomicAdd(total_water_loss, loss);
                    }
                } else {
                    int neigh_idx = new_row * columns + new_col;
                    neighbor_height = ground[neigh_idx] + FLOATING(water_level[neigh_idx]);
                    if (current_height >= neighbor_height) {
                        int spill_idx = neigh_idx * CONTIGUOUS_CELLS + cell_pos;
                        spillage_from_neigh[spill_idx] = proportion * (current_height - neighbor_height);
                    }
                }
            }
        }
    }
}



/*
 * Main compute function
 */
extern "C" void do_compute(struct parameters *p, struct results *r) {
    int rows = p->rows, columns = p->columns;
    int *minute = &r->minute;

    /* 2. Start global timer */
    CUDA_CHECK_FUNCTION(cudaSetDevice(0));

    /*
     *
     * Allocate memory and call kernels in this function.
     * Ensure all debug and animation code works in your final version.
     *
     */

    /* Memory allocation */

    int *water_level;           // Level of water on each cell (fixed precision)
    float *ground;              // Ground height

    float *d_ground = NULL;
    int *d_water_level = NULL;
    float *d_spillage_flag = NULL;
    float *d_spillage_level = NULL;
    float *d_spillage_from_neigh = NULL;
    unsigned long long *d_total_water_loss = NULL;
    unsigned long long *d_total_rain = NULL;
    int *d_max_spillage_bits = NULL;

    ground = p->ground;
    water_level = (int *)malloc(sizeof(int) * (size_t)rows * (size_t)columns);

    if (water_level == NULL) {
        fprintf(stderr, "-- Error allocating ground and rain structures for size: %d x %d \n", rows, columns);
        exit(EXIT_FAILURE);
    }

    size_t cells_size_int = sizeof(int) * (size_t)rows * (size_t)columns;
    size_t cells_size_float = sizeof(float) * (size_t)rows * (size_t)columns;
    size_t spill_neigh_size = sizeof(float) * (size_t)rows * (size_t)columns * (size_t)CONTIGUOUS_CELLS;

    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_ground, cells_size_float));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_water_level, cells_size_int));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_spillage_flag, cells_size_float));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_spillage_level, cells_size_float));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_spillage_from_neigh, spill_neigh_size));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_total_water_loss, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_total_rain, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_max_spillage_bits, sizeof(int)));

    CUDA_CHECK_FUNCTION(cudaMemcpy(d_ground, ground, cells_size_float, cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemset(d_water_level, 0, cells_size_int));
    CUDA_CHECK_FUNCTION(cudaMemset(d_total_water_loss, 0, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMemset(d_total_rain, 0, sizeof(unsigned long long)));

    CUDA_CHECK_FUNCTION(cudaMemset(d_max_spillage_bits, 0, sizeof(int)));

    /* Ground generation and initialization of other structures */
    int row_pos, col_pos;

#ifdef DEBUG
    print_matrix(PRECISION_FLOAT, rows, columns, ground, "Ground heights");
#ifndef ANIMATION
    print_clouds(p->num_clouds, p->clouds);
#endif
#endif

    double max_spillage_iter = DBL_MAX;

    /* Opt 3: Precompute trig, cos/sin are constant per cloud, avoid recomputing each iteration */
    float *cloud_dx = (float *)malloc(sizeof(float) * p->num_clouds);
    float *cloud_dy = (float *)malloc(sizeof(float) * p->num_clouds);
    for (int c = 0; c < p->num_clouds; c++) {
        cloud_dx[c] = (p->clouds[c].speed / 60.0f) * cosf(p->clouds[c].angle * (float)M_PI / 180.0f);
        cloud_dy[c] = (p->clouds[c].speed / 60.0f) * sinf(p->clouds[c].angle * (float)M_PI / 180.0f);
    }

    /* Prepare to measure runtime */
    r->runtime = get_time();

    /* Opt 4: Hoist invariants, these values are constant across all iterations (compiler likely already does this) */
    int total_cells = rows * columns;
    int block_size = 256;
    int grid_size = (total_cells + block_size - 1) / block_size;

    /* Flood simulation */
    for (*minute = 0; *minute < p->num_minutes && max_spillage_iter > p->threshold; (*minute)++) {

        /* Step 1.1: Clouds movement, replaces per-iteration cos/sin with simple addition */
        for (int cloud = 0; cloud < p->num_clouds; cloud++) {
            p->clouds[cloud].x += cloud_dx[cloud];
            p->clouds[cloud].y += cloud_dy[cloud];
        }

#ifdef DEBUG
#ifndef ANIMATION
        print_clouds(p->num_clouds, p->clouds);
#endif
#endif

        /* Step 1.2: Rainfall */
        for (int cloud = 0; cloud < p->num_clouds; cloud++) {
            Cloud_t c_cloud = p->clouds[cloud];
            float row_start = COORD_SCEN2MAT_Y(MAX(0, c_cloud.y - c_cloud.radius));
            float row_end = COORD_SCEN2MAT_Y(MIN(c_cloud.y + c_cloud.radius, SCENARIO_SIZE));
            float col_start = COORD_SCEN2MAT_X(MAX(0, c_cloud.x - c_cloud.radius));
            float col_end = COORD_SCEN2MAT_X(MIN(c_cloud.x + c_cloud.radius, SCENARIO_SIZE));

            int row_start_idx = MAX(0, (int)row_start);
            int col_start_idx = MAX(0, (int)col_start);
            int row_count = (int)ceilf(row_end - row_start);
            int col_count = (int)ceilf(col_end - col_start);

            row_count = MIN(rows - row_start_idx, row_count);
            col_count = MIN(columns - col_start_idx, col_count);

            if (row_count <= 0 || col_count <= 0)
                continue;

            int rain_cells = row_count * col_count;
            int rain_grid_size = (rain_cells + block_size - 1) / block_size;
            rainfall_kernel<<<rain_grid_size, block_size>>>(rows, columns, row_start, row_count, col_start,
                                                           col_count, c_cloud, p->ex_factor, d_water_level,
                                                           d_total_rain);
            CUDA_CHECK_KERNEL();
        }

#ifdef DEBUG
        CUDA_CHECK_FUNCTION(cudaMemcpy(water_level, d_water_level, cells_size_int, cudaMemcpyDeviceToHost));
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after rain");
#endif

        /* Step 2: Compute water spillage to neighbor cells */
        step2_spillage_kernel<<<grid_size, block_size>>>(rows, columns, d_ground, d_water_level, d_spillage_flag,
                                                         d_spillage_level, d_spillage_from_neigh,
                                                         d_total_water_loss);
        CUDA_CHECK_KERNEL();

        /* Step 3: Propagation of previuosly computer water spillage to/from neighbors */
        max_spillage_iter = 0.0;

        /* Opt 2: Async memset, avoids host-device sync before propagation kernel */
        CUDA_CHECK_FUNCTION(cudaMemsetAsync(d_max_spillage_bits, 0, sizeof(int), 0));
        step3_propagation_kernel<<<grid_size, block_size>>>(rows, columns, d_water_level, d_spillage_flag,
                                    d_spillage_level, d_spillage_from_neigh,
                                    d_max_spillage_bits);
        CUDA_CHECK_KERNEL();

        int max_spillage_bits = 0;
        CUDA_CHECK_FUNCTION(cudaMemcpy(&max_spillage_bits, d_max_spillage_bits, sizeof(int), cudaMemcpyDeviceToHost));
        {
            float max_spillage_iter_float = 0.0f;
            memcpy(&max_spillage_iter_float, &max_spillage_bits, sizeof(float));
            max_spillage_iter = max_spillage_iter_float;
        }

        if (max_spillage_iter > r->max_spillage_scenario) {
            r->max_spillage_scenario = max_spillage_iter;
            r->max_spillage_minute = *minute;
        }

#ifdef DEBUG
#ifndef ANIMATION
        CUDA_CHECK_FUNCTION(cudaMemcpy(water_level, d_water_level, cells_size_int, cudaMemcpyDeviceToHost));
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after spillage");
#endif
#endif

    }

    r->runtime = get_time() - r->runtime;

    if (p->final_matrix) {
        CUDA_CHECK_FUNCTION(cudaMemcpy(water_level, d_water_level, cells_size_int, cudaMemcpyDeviceToHost));
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after spillage");
    }

    CUDA_CHECK_FUNCTION(cudaMemcpy(water_level, d_water_level, cells_size_int, cudaMemcpyDeviceToHost));

    /* Statistics: Total remaining water and maximum amount of water in a cell */
    r->max_water_scenario = 0.0;
    for (row_pos = 0; row_pos < rows; row_pos++) {
        for (col_pos = 0; col_pos < columns; col_pos++) {
            if (FLOATING(accessMat(water_level, row_pos, col_pos)) > r->max_water_scenario)
                r->max_water_scenario = FLOATING(accessMat(water_level, row_pos, col_pos));
            r->total_water += accessMat(water_level, row_pos, col_pos);
        }
    }

    /* Free resources */
    free(water_level);
    free(cloud_dx);
    free(cloud_dy);

    unsigned long long total_water_loss_host = 0;
    CUDA_CHECK_FUNCTION(
        cudaMemcpy(&total_water_loss_host, d_total_water_loss, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    r->total_water_loss = (long)total_water_loss_host;
    unsigned long long total_rain_host = 0;
    CUDA_CHECK_FUNCTION(cudaMemcpy(&total_rain_host, d_total_rain, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    r->total_rain = (long)total_rain_host;

    CUDA_CHECK_FUNCTION(cudaFree(d_ground));
    CUDA_CHECK_FUNCTION(cudaFree(d_water_level));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_flag));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_level));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_from_neigh));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_water_loss));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_rain));
    CUDA_CHECK_FUNCTION(cudaFree(d_max_spillage_bits));

}
