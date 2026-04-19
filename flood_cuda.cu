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

/* spillage_level > 0 replaces the separate spillage_flag array, saving one
   global read + write per cell per iteration */
__global__ void step3_propagation_kernel(int rows,
                                         int columns,
                                         int *water_level,
                                         float *spillage_level,
                                         float *spill_neigh_0,
                                         float *spill_neigh_1,
                                         float *spill_neigh_2,
                                         float *spill_neigh_3,
                                         int *max_spillage_bits) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = rows * columns;
    if (idx >= total_cells)
        return;

    float sl = spillage_level[idx];
    if (sl > 0.0f) {
        float current_spillage = sl / SPILLAGE_FACTOR;
        water_level[idx] -= FIXED(current_spillage);
        atomicMax(max_spillage_bits, __float_as_int(current_spillage));
        spillage_level[idx] = 0.0f;
    }

    /* SoA: consecutive threads now read consecutive addresses per array */
    water_level[idx] += FIXED(spill_neigh_0[idx] / SPILLAGE_FACTOR);
    water_level[idx] += FIXED(spill_neigh_1[idx] / SPILLAGE_FACTOR);
    water_level[idx] += FIXED(spill_neigh_2[idx] / SPILLAGE_FACTOR);
    water_level[idx] += FIXED(spill_neigh_3[idx] / SPILLAGE_FACTOR);

    /* Reset after read: eliminates separate reset kernel launch per iteration */
    spill_neigh_0[idx] = 0.0f;
    spill_neigh_1[idx] = 0.0f;
    spill_neigh_2[idx] = 0.0f;
    spill_neigh_3[idx] = 0.0f;
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

/* Shared memory tiling: load ground + water_level into 18x18 tile (16x16 + 1-cell border)
   so neighbor lookups read from fast on-chip SRAM instead of global memory.
   Border cells are the 1-cell-wide border loaded so edge threads can read their neighbors. */
#define TILE_W 16
#define TILE_H 16
#define STILE_W (TILE_W + 2)
#define STILE_H (TILE_H + 2)

__global__ void step2_spillage_kernel(int rows,
                                      int columns,
                                      const float *ground,
                                      const int *water_level,
                                      float *spillage_level,
                                      float *spill_neigh_0,
                                      float *spill_neigh_1,
                                      float *spill_neigh_2,
                                      float *spill_neigh_3,
                                      unsigned long long *total_water_loss) {
    __shared__ float s_ground[STILE_H][STILE_W];
    __shared__ int   s_water[STILE_H][STILE_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_W + tx;
    int row = blockIdx.y * TILE_H + ty;
    int idx = row * columns + col;
    int valid = (row < rows && col < columns);

    /* Inner cell: each thread loads its own cell at (ty+1, tx+1) */
    int sx = tx + 1;
    int sy = ty + 1;
    if (valid) {
        s_ground[sy][sx] = ground[idx];
        s_water[sy][sx]  = water_level[idx];
    } else {
        s_ground[sy][sx] = 0.0f;
        s_water[sy][sx]  = 0;
    }

    /* Border cells: top row */
    if (ty == 0) {
        int hrow = row - 1;
        int hcol = col;
        if (hrow >= 0 && hcol < columns) {
            int hidx = hrow * columns + hcol;
            s_ground[0][sx] = ground[hidx];
            s_water[0][sx]  = water_level[hidx];
        } else {
            s_ground[0][sx] = valid ? ground[idx] : 0.0f;
            s_water[0][sx]  = 0;
        }
    }
    /* Border cells: bottom row */
    if (ty == TILE_H - 1) {
        int hrow = row + 1;
        int hcol = col;
        if (hrow < rows && hcol < columns) {
            int hidx = hrow * columns + hcol;
            s_ground[STILE_H - 1][sx] = ground[hidx];
            s_water[STILE_H - 1][sx]  = water_level[hidx];
        } else {
            s_ground[STILE_H - 1][sx] = valid ? ground[idx] : 0.0f;
            s_water[STILE_H - 1][sx]  = 0;
        }
    }
    /* Border cells: left column */
    if (tx == 0) {
        int hrow = row;
        int hcol = col - 1;
        if (hcol >= 0 && hrow < rows) {
            int hidx = hrow * columns + hcol;
            s_ground[sy][0] = ground[hidx];
            s_water[sy][0]  = water_level[hidx];
        } else {
            s_ground[sy][0] = valid ? ground[idx] : 0.0f;
            s_water[sy][0]  = 0;
        }
    }
    /* Border cells: right column */
    if (tx == TILE_W - 1) {
        int hrow = row;
        int hcol = col + 1;
        if (hcol < columns && hrow < rows) {
            int hidx = hrow * columns + hcol;
            s_ground[sy][STILE_W - 1] = ground[hidx];
            s_water[sy][STILE_W - 1]  = water_level[hidx];
        } else {
            s_ground[sy][STILE_W - 1] = valid ? ground[idx] : 0.0f;
            s_water[sy][STILE_W - 1]  = 0;
        }
    }

    /* All threads must finish loading shared memory before any thread reads it.
       Must be before the early-return checks to avoid deadlock. */
    __syncthreads();

    /* Threads past grid edges (when grid size isn't divisible by tile size) */
    if (!valid)
        return;
    if (water_level[idx] <= 0)
        return;

    float sum_diff = 0.0f;
    float my_spillage_level = 0.0f;
    float current_height = s_ground[sy][sx] + FLOATING(s_water[sy][sx]);

    /* Cache neighbor heights and boundary flags so the second loop can reuse them */
    float neigh_h[CONTIGUOUS_CELLS];
    /* True if neighbor is outside grid (boundary spillage becomes water loss) */
    int neigh_is_boundary[CONTIGUOUS_CELLS];

    for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
        int new_row = row + displacements[cell_pos][0];
        int new_col = col + displacements[cell_pos][1];
        neigh_is_boundary[cell_pos] = (new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns);

        if (neigh_is_boundary[cell_pos]) {
            neigh_h[cell_pos] = s_ground[sy][sx];
        } else {
            int ny = sy + displacements[cell_pos][0];
            int nx = sx + displacements[cell_pos][1];
            neigh_h[cell_pos] = s_ground[ny][nx] + FLOATING(s_water[ny][nx]);
        }

        if (current_height >= neigh_h[cell_pos]) {
            float height_diff = current_height - neigh_h[cell_pos];
            sum_diff += height_diff;
            my_spillage_level = MAX(my_spillage_level, height_diff);
        }
    }

    my_spillage_level = MIN(FLOATING(s_water[sy][sx]), my_spillage_level);

    if (sum_diff > 0.0f) {
        float proportion = my_spillage_level / sum_diff;
        if (proportion > 1e-8f) {
            spillage_level[idx] = my_spillage_level;

            float *spill_neigh[CONTIGUOUS_CELLS] = {
                spill_neigh_0, spill_neigh_1, spill_neigh_2, spill_neigh_3
            };
            for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
                if (neigh_is_boundary[cell_pos]) {
                    if (current_height >= neigh_h[cell_pos]) {
                        unsigned long long loss =
                            (unsigned long long)FIXED(proportion * (current_height - neigh_h[cell_pos]) / 2.0f);
                        atomicAdd(total_water_loss, loss);
                    }
                } else {
                    if (current_height >= neigh_h[cell_pos]) {
                        int new_row = row + displacements[cell_pos][0];
                        int new_col = col + displacements[cell_pos][1];
                        int neigh_idx = new_row * columns + new_col;
                        spill_neigh[cell_pos][neigh_idx] =
                            proportion * (current_height - neigh_h[cell_pos]);
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
    float *d_spillage_level = NULL;
    float *d_spill_neigh[CONTIGUOUS_CELLS] = {NULL, NULL, NULL, NULL};
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

    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_ground, cells_size_float));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_water_level, cells_size_int));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_spillage_level, cells_size_float));
    for (int i = 0; i < CONTIGUOUS_CELLS; i++)
        CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_spill_neigh[i], cells_size_float));
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

    /* Optimization precompute trigonometry: cos/sin are constant per cloud.This avoids 
    recomputing them at each iteration */
    float *cloud_dx = (float *)malloc(sizeof(float) * p->num_clouds);
    float *cloud_dy = (float *)malloc(sizeof(float) * p->num_clouds);
    for (int c = 0; c < p->num_clouds; c++) {
        cloud_dx[c] = (p->clouds[c].speed / 60.0f) * cosf(p->clouds[c].angle * (float)M_PI / 180.0f);
        cloud_dy[c] = (p->clouds[c].speed / 60.0f) * sinf(p->clouds[c].angle * (float)M_PI / 180.0f);
    }

    /* Prepare to measure runtime */
    r->runtime = get_time();

    int total_cells = rows * columns;
    int block_size = 256;
    int grid_size = (total_cells + block_size - 1) / block_size;
    dim3 block2d(TILE_W, TILE_H);
    dim3 grid2d((columns + TILE_W - 1) / TILE_W, (rows + TILE_H - 1) / TILE_H);

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

        /* Step 2: Compute water spillage to neighbor cells (2D blocks with shared memory tiling) */
        step2_spillage_kernel<<<grid2d, block2d>>>(rows, columns, d_ground, d_water_level,
                                                   d_spillage_level,
                                                   d_spill_neigh[0], d_spill_neigh[1],
                                                   d_spill_neigh[2], d_spill_neigh[3],
                                                   d_total_water_loss);
        CUDA_CHECK_KERNEL();

        /* Step 3: Propagation of previuosly computer water spillage to/from neighbors */
        max_spillage_iter = 0.0;

        /* Optimization Async memset: avoids host-device sync before 
         *propagation kernel */
        CUDA_CHECK_FUNCTION(cudaMemsetAsync(d_max_spillage_bits, 0, sizeof(int), 0));
        step3_propagation_kernel<<<grid_size, block_size>>>(rows, columns, d_water_level,
                                    d_spillage_level,
                                    d_spill_neigh[0], d_spill_neigh[1],
                                    d_spill_neigh[2], d_spill_neigh[3],
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
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_level));
    for (int i = 0; i < CONTIGUOUS_CELLS; i++)
        CUDA_CHECK_FUNCTION(cudaFree(d_spill_neigh[i]));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_water_loss));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_rain));
    CUDA_CHECK_FUNCTION(cudaFree(d_max_spillage_bits));

}
