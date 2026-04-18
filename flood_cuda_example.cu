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
                                         const float *spillage_flag,
                                         const float *spillage_level,
                                         const float *spillage_from_neigh,
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
}

__global__ void reset_spillage_kernel(int rows,
                                      int columns,
                                      float *spillage_flag,
                                      float *spillage_level,
                                      float *spillage_from_neigh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = rows * columns;
    if (idx >= total_cells)
        return;

    spillage_flag[idx] = 0.0f;
    spillage_level[idx] = 0.0f;

    int base = idx * CONTIGUOUS_CELLS;
    for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
        spillage_from_neigh[base + cell_pos] = 0.0f;
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
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

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
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_max_spillage_bits, sizeof(int)));

    CUDA_CHECK_FUNCTION(cudaMemcpy(d_ground, ground, cells_size_float, cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemset(d_water_level, 0, cells_size_int));
    CUDA_CHECK_FUNCTION(cudaMemset(d_total_water_loss, 0, sizeof(unsigned long long)));

    /* Ground generation and initialization of other structures */
    int row_pos, col_pos;
    for (row_pos = 0; row_pos < rows; row_pos++) {
        for (col_pos = 0; col_pos < columns; col_pos++) {
            accessMat(water_level, row_pos, col_pos) = 0;
        }
    }

#ifdef DEBUG
    print_matrix(PRECISION_FLOAT, rows, columns, ground, "Ground heights");
#ifndef ANIMATION
    print_clouds(p->num_clouds, p->clouds);
#endif
#endif

    double max_spillage_iter = DBL_MAX;

    /* Prepare to measure runtime */
    r->runtime = get_time();

    /* Flood simulation */
    for (*minute = 0; *minute < p->num_minutes && max_spillage_iter > p->threshold; (*minute)++) {
        int total_cells = rows * columns;
        int block_size = 256;
        int grid_size = (total_cells + block_size - 1) / block_size;

        /* Step 1.1: Clouds movement */
        for (int cloud = 0; cloud < p->num_clouds; cloud++) {
            // Calculate new position (x are rows, y are columns)
            Cloud_t *c_cloud = &p->clouds[cloud];
            float km_minute = c_cloud->speed / 60;
            c_cloud->x += km_minute * cos(c_cloud->angle * M_PI / 180.0);
            c_cloud->y += km_minute * sin(c_cloud->angle * M_PI / 180.0);
        }

#ifdef DEBUG
#ifndef ANIMATION
        print_clouds(p->num_clouds, p->clouds);
#endif
#endif

        /* Step 1.2: Rainfall */
        for (int cloud = 0; cloud < p->num_clouds; cloud++) {
            Cloud_t c_cloud = p->clouds[cloud];
            // Compute the bounding box area of the cloud
            float row_start = COORD_SCEN2MAT_Y(MAX(0, c_cloud.y - c_cloud.radius));
            float row_end = COORD_SCEN2MAT_Y(MIN(c_cloud.y + c_cloud.radius, SCENARIO_SIZE));
            float col_start = COORD_SCEN2MAT_X(MAX(0, c_cloud.x - c_cloud.radius));
            float col_end = COORD_SCEN2MAT_X(MIN(c_cloud.x + c_cloud.radius, SCENARIO_SIZE));
            float distance;

            // Add rain to the ground water level
            float row_pos, col_pos;
            for (row_pos = row_start; row_pos < row_end; row_pos++) {
                for (col_pos = col_start; col_pos < col_end; col_pos++) {
                    float x_pos = COORD_MAT2SCEN_X(col_pos);
                    float y_pos = COORD_MAT2SCEN_Y(row_pos);
                    distance =
                        sqrt((x_pos - c_cloud.x) * (x_pos - c_cloud.x) + (y_pos - c_cloud.y) * (y_pos - c_cloud.y));
                    if (distance < c_cloud.radius) {
                        float rain = p->ex_factor *
                                     MAX(0, c_cloud.intensity - distance / c_cloud.radius * sqrt(c_cloud.intensity));
                        float meters_per_minute = rain / 1000 / 60;
                        accessMat(water_level, row_pos, col_pos) += FIXED(meters_per_minute);
                        r->total_rain += FIXED(meters_per_minute);
                    }
                }
            }
        }

#ifdef DEBUG
        CUDA_CHECK_FUNCTION(cudaMemcpy(water_level, d_water_level, cells_size_int, cudaMemcpyDeviceToHost));
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after rain");
#endif

        /* Step 2: Compute water spillage to neighbor cells */
        CUDA_CHECK_FUNCTION(cudaMemcpy(d_water_level, water_level, cells_size_int, cudaMemcpyHostToDevice));
        reset_spillage_kernel<<<grid_size, block_size>>>(rows, columns, d_spillage_flag, d_spillage_level,
                                                         d_spillage_from_neigh);
        CUDA_CHECK_KERNEL();

        step2_spillage_kernel<<<grid_size, block_size>>>(rows, columns, d_ground, d_water_level, d_spillage_flag,
                                                         d_spillage_level, d_spillage_from_neigh,
                                                         d_total_water_loss);
        CUDA_CHECK_KERNEL();

        /* Step 3: Propagation of previuosly computer water spillage to/from neighbors */
        max_spillage_iter = 0.0;

        CUDA_CHECK_FUNCTION(cudaMemset(d_max_spillage_bits, 0, sizeof(int)));
        step3_propagation_kernel<<<grid_size, block_size>>>(rows, columns, d_water_level, d_spillage_flag,
                                    d_spillage_level, d_spillage_from_neigh,
                                    d_max_spillage_bits);
        CUDA_CHECK_KERNEL();

        CUDA_CHECK_FUNCTION(cudaMemcpy(water_level, d_water_level, cells_size_int, cudaMemcpyDeviceToHost));

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
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after spillage");
#endif
#endif

    }

    cudaDeviceSynchronize();

    r->runtime = get_time() - r->runtime;

    if (p->final_matrix) {
        CUDA_CHECK_FUNCTION(cudaMemcpy(water_level, d_water_level, cells_size_int, cudaMemcpyDeviceToHost));
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after spillage");
    }

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

    unsigned long long total_water_loss_host = 0;
    CUDA_CHECK_FUNCTION(
        cudaMemcpy(&total_water_loss_host, d_total_water_loss, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    r->total_water_loss = (long)total_water_loss_host;

    CUDA_CHECK_FUNCTION(cudaFree(d_ground));
    CUDA_CHECK_FUNCTION(cudaFree(d_water_level));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_flag));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_level));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_from_neigh));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_water_loss));
    CUDA_CHECK_FUNCTION(cudaFree(d_max_spillage_bits));

    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
}
