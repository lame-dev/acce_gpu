/*
 * Simulation of rainwater flooding
 * CUDA Parallel Implementation - CORRECTED VERSION
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

// Thread block size for 2D kernels
#define BLOCK_SIZE 16

/*
 * CUDA Kernel: Initialize water level and auxiliary matrices
 */
__global__ void kernel_initialize(int *water_level, float *spillage_flag, float *spillage_level,
                                   float *spillage_from_neigh, int rows, int columns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < columns) {
        int idx = row * columns + col;
        water_level[idx] = 0;
        spillage_flag[idx] = 0.0f;
        spillage_level[idx] = 0.0f;

        // Initialize spillage_from_neigh for all 4 directions
        for (int d = 0; d < CONTIGUOUS_CELLS; d++) {
            spillage_from_neigh[idx * CONTIGUOUS_CELLS + d] = 0.0f;
        }
    }
}

/*
 * CUDA Kernel: Add rainfall from clouds to water level
 */
__global__ void kernel_rainfall(int *water_level, Cloud_t *clouds, int num_clouds, int rows,
                                 int columns, float ex_factor, long *total_rain_global) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= columns)
        return;

    int idx = row * columns + col;
    long local_rain = 0;
    
    float x_pos = COORD_MAT2SCEN_X(col);
    float y_pos = COORD_MAT2SCEN_Y(row);

    // Process each cloud
    for (int cloud = 0; cloud < num_clouds; cloud++) {
        Cloud_t c_cloud = clouds[cloud];
        
        float distance = sqrtf((x_pos - c_cloud.x) * (x_pos - c_cloud.x) + 
                               (y_pos - c_cloud.y) * (y_pos - c_cloud.y));

        if (distance < c_cloud.radius) {
            float rain = ex_factor * fmaxf(0.0f, c_cloud.intensity - 
                                          distance / c_cloud.radius * sqrtf(c_cloud.intensity));
            float meters_per_minute = rain / 1000.0f / 60.0f;
            int rain_fixed = FIXED(meters_per_minute);
            atomicAdd(&water_level[idx], rain_fixed);
            local_rain += rain_fixed;
        }
    }

    // Accumulate total rain
    if (local_rain > 0) {
        atomicAdd((unsigned long long *)total_rain_global, (unsigned long long)local_rain);
    }
}

/*
 * CUDA Kernel: Compute water spillage to neighbor cells
 */
__global__ void kernel_compute_spillage(int *water_level, float *ground, float *spillage_flag, float *spillage_level,
                                         float *spillage_from_neigh, int rows, int columns,
                                         long *total_water_loss_global) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= columns)
        return;

    int idx = row * columns + col;

    if (water_level[idx] <= 0) {
        spillage_flag[idx] = 0.0f;
        spillage_level[idx] = 0.0f;
        return;
    }

    float sum_diff = 0.0f;
    float my_spillage_level = 0.0f;

    // Current cell height (ground + water)
    float current_height = ground[idx] + FLOATING(water_level[idx]);

    // First pass: compute differences and find max spillage level
    for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
        int new_row = row + displacements[cell_pos][0];
        int new_col = col + displacements[cell_pos][1];

        float neighbor_height;

        // Check if the new position is within the matrix boundaries
        if (new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns) {
            // Out of borders: Same height as the current cell's ground (without water)
            neighbor_height = ground[idx];
        } else {
            // Neighbor cell: Ground height + water level
            int neighbor_idx = new_row * columns + new_col;
            neighbor_height = ground[neighbor_idx] + FLOATING(water_level[neighbor_idx]);
        }

        // Compute level differences
        if (current_height > neighbor_height) {
            float height_diff = current_height - neighbor_height;
            sum_diff += height_diff;
            my_spillage_level = fmaxf(my_spillage_level, height_diff);
        }
    }

    my_spillage_level = fminf(FLOATING(water_level[idx]), my_spillage_level);

    // Compute proportion of spillage to each neighbor
    if (sum_diff > 0.0f && my_spillage_level > 0.0f) {
        float proportion = my_spillage_level / sum_diff;

        // If proportion is significative, spillage
        if (proportion > 1e-8f) {
            spillage_flag[idx] = 1.0f;
            spillage_level[idx] = my_spillage_level;

            long local_water_loss = 0;

            // Second pass: distribute spillage
            for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
                int new_row = row + displacements[cell_pos][0];
                int new_col = col + displacements[cell_pos][1];

                float neighbor_height;

                // Check if the new position is within the matrix boundaries
                if (new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns) {
                    // Spillage out of the borders: Water loss
                    neighbor_height = ground[idx];
                    if (current_height > neighbor_height) {
                        local_water_loss += FIXED(proportion * (current_height - neighbor_height) / 2.0f);
                    }
                } else {
                    // Spillage to a neighbor cell
                    int neighbor_idx = new_row * columns + new_col;
                    neighbor_height = ground[neighbor_idx] + FLOATING(water_level[neighbor_idx]);
                    if (current_height > neighbor_height) {
                        spillage_from_neigh[neighbor_idx * CONTIGUOUS_CELLS + cell_pos] =
                            proportion * (current_height - neighbor_height);
                    }
                }
            }

            if (local_water_loss > 0) {
                atomicAdd((unsigned long long *)total_water_loss_global, (unsigned long long)local_water_loss);
            }
        }
    } else {
        spillage_flag[idx] = 0.0f;
        spillage_level[idx] = 0.0f;
    }
}

/*
 * CUDA Kernel: Propagate water spillage and update water levels
 */
__global__ void kernel_propagate_spillage(int *water_level, float *spillage_flag, float *spillage_level,
                                           float *spillage_from_neigh, int rows, int columns,
                                           float *max_spillage_iter_cell) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= columns)
        return;

    int idx = row * columns + col;
    float local_max = 0.0f;

    // If the cell has spillage, remove it from water level
    if (spillage_flag[idx] == 1.0f) {
        float spillage_amount = spillage_level[idx] / SPILLAGE_FACTOR;
        atomicAdd(&water_level[idx], -FIXED(spillage_amount));
        local_max = spillage_amount;
    }

    // Accumulate spillage from neighbors
    for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
        float spillage = spillage_from_neigh[idx * CONTIGUOUS_CELLS + cell_pos];
        if (spillage > 0.0f) {
            atomicAdd(&water_level[idx], FIXED(spillage / SPILLAGE_FACTOR));
        }
    }
    
    // Store the local max for reduction
    max_spillage_iter_cell[idx] = local_max;
}

/*
 * CUDA Kernel: Reduce max values and update global maximums
 */
__global__ void kernel_reduce_max_spillage(float *max_spillage_iter_cell, int size, 
                                            double *max_spillage_iter, double *max_spillage_scenario,
                                            int *max_spillage_minute, int minute) {
    __shared__ float shared_max[BLOCK_SIZE * BLOCK_SIZE];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int idx = blockIdx.x * blockDim.x * blockDim.y + tid;
    
    // Load data into shared memory
    if (idx < size) {
        shared_max[tid] = max_spillage_iter_cell[idx];
    } else {
        shared_max[tid] = 0.0f;
    }
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    // Thread 0 writes result
    if (tid == 0) {
        float block_max = shared_max[0];
        if (block_max > 0.0f) {
            double block_max_double = (double)block_max;
            
            // Update iteration max using atomic operations
            unsigned long long *max_iter_ptr = (unsigned long long *)max_spillage_iter;
            unsigned long long old_val = *max_iter_ptr;
            double old_max = __longlong_as_double(old_val);
            
            if (block_max_double > old_max) {
                unsigned long long new_val = __double_as_longlong(block_max_double);
                atomicCAS(max_iter_ptr, old_val, new_val);
            }
            
            // Update scenario max - use lock to ensure minute is updated correctly
            unsigned long long *max_scen_ptr = (unsigned long long *)max_spillage_scenario;
            old_val = *max_scen_ptr;
            old_max = __longlong_as_double(old_val);
            
            if (block_max_double > old_max) {
                unsigned long long new_val = __double_as_longlong(block_max_double);
                unsigned long long ret = atomicCAS(max_scen_ptr, old_val, new_val);
                if (ret == old_val) {
                    // Only update minute if we successfully updated the max
                    *max_spillage_minute = minute;
                }
            }
        }
    }
}

/*
 * CUDA Kernel: Reset ancillary structures
 */
__global__ void kernel_reset_structures(float *spillage_flag, float *spillage_level, float *spillage_from_neigh,
                                         int rows, int columns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= columns)
        return;

    int idx = row * columns + col;

    spillage_flag[idx] = 0.0f;
    spillage_level[idx] = 0.0f;

    for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
        spillage_from_neigh[idx * CONTIGUOUS_CELLS + cell_pos] = 0.0f;
    }
}

/*
 * CUDA Kernel: Compute final statistics (max water and total water)
 */
__global__ void kernel_compute_statistics(int *water_level, int rows, int columns, float *max_water_scenario,
                                           long *total_water_global) {
    __shared__ float shared_max[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ long shared_sum[BLOCK_SIZE * BLOCK_SIZE];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int idx = blockIdx.x * blockDim.x * blockDim.y + tid;
    int size = rows * columns;
    
    // Initialize shared memory
    shared_max[tid] = 0.0f;
    shared_sum[tid] = 0;
    
    if (idx < size) {
        int water_val = water_level[idx];
        float water = FLOATING(water_val);
        shared_max[tid] = water;
        shared_sum[tid] = water_val;
    }
    __syncthreads();
    
    // Reduction
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes result
    if (tid == 0) {
        // Update max water using atomic operations
        if (shared_max[0] > 0.0f) {
            unsigned int *max_water_ptr = (unsigned int *)max_water_scenario;
            unsigned int old_val = *max_water_ptr;
            float old_max = __uint_as_float(old_val);
            
            if (shared_max[0] > old_max) {
                unsigned int new_val = __float_as_uint(shared_max[0]);
                atomicCAS(max_water_ptr, old_val, new_val);
            }
        }
        
        // Add to total water
        if (shared_sum[0] > 0) {
            atomicAdd((unsigned long long *)total_water_global, (unsigned long long)shared_sum[0]);
        }
    }
}

/*
 * Main compute function
 */
extern "C" void do_compute(struct parameters *p, struct results *r) {
    int rows = p->rows, columns = p->columns;
    int *minute = &r->minute;

    /* Set CUDA device */
    CUDA_CHECK_FUNCTION(cudaSetDevice(0));

    /* Define grid and block dimensions */
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((columns + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    /* Allocate host memory */
    int *h_water_level;
    float *h_ground;
    float *h_spillage_flag;
    float *h_spillage_level;
    float *h_spillage_from_neigh;

    h_ground = p->ground;
    h_water_level = (int *)malloc(sizeof(int) * rows * columns);
    h_spillage_flag = (float *)malloc(sizeof(float) * rows * columns);
    h_spillage_level = (float *)malloc(sizeof(float) * rows * columns);
    h_spillage_from_neigh = (float *)malloc(sizeof(float) * rows * columns * CONTIGUOUS_CELLS);

    if (h_water_level == NULL || h_spillage_flag == NULL || h_spillage_level == NULL ||
        h_spillage_from_neigh == NULL) {
        fprintf(stderr, "-- Error allocating host memory for size: %d x %d \n", rows, columns);
        exit(EXIT_FAILURE);
    }

    /* Allocate device memory */
    int *d_water_level;
    float *d_ground;
    float *d_spillage_flag;
    float *d_spillage_level;
    float *d_spillage_from_neigh;
    float *d_max_spillage_iter_cell;
    Cloud_t *d_clouds;
    long *d_total_rain;
    long *d_total_water_loss;
    long *d_total_water;
    double *d_max_spillage_iter;
    double *d_max_spillage_scenario;
    int *d_max_spillage_minute;
    float *d_max_water_scenario;

    size_t size_grid = sizeof(int) * rows * columns;
    size_t size_grid_float = sizeof(float) * rows * columns;
    size_t size_spillage_neigh = sizeof(float) * rows * columns * CONTIGUOUS_CELLS;
    size_t size_clouds = sizeof(Cloud_t) * p->num_clouds;

    CUDA_CHECK_FUNCTION(cudaMalloc(&d_water_level, size_grid));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_ground, size_grid_float));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_spillage_flag, size_grid_float));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_spillage_level, size_grid_float));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_spillage_from_neigh, size_spillage_neigh));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_max_spillage_iter_cell, size_grid_float));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_clouds, size_clouds));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_total_rain, sizeof(long)));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_total_water_loss, sizeof(long)));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_total_water, sizeof(long)));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_max_spillage_iter, sizeof(double)));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_max_spillage_scenario, sizeof(double)));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_max_spillage_minute, sizeof(int)));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_max_water_scenario, sizeof(float)));

    /* Copy ground data to device */
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_ground, h_ground, size_grid_float, cudaMemcpyHostToDevice));

    /* Initialize device variables */
    long h_zero = 0;
    double h_zero_double = 0.0;
    int h_zero_int = 0;
    float h_zero_float = 0.0f;
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_total_rain, &h_zero, sizeof(long), cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_total_water_loss, &h_zero, sizeof(long), cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_total_water, &h_zero, sizeof(long), cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_max_spillage_scenario, &h_zero_double, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_max_spillage_minute, &h_zero_int, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_max_water_scenario, &h_zero_float, sizeof(float), cudaMemcpyHostToDevice));

    /* Initialize matrices on GPU */
    kernel_initialize<<<gridDim, blockDim>>>(d_water_level, d_spillage_flag, d_spillage_level, d_spillage_from_neigh,
                                              rows, columns);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

#ifdef DEBUG
    print_matrix(PRECISION_FLOAT, rows, columns, h_ground, "Ground heights");
#ifndef ANIMATION
    print_clouds(p->num_clouds, p->clouds);
#endif
#endif

    /* Prepare to measure runtime */
    r->runtime = get_time();

    // Calculate grid for reduction kernel
    int reduction_grid_size = (rows * columns + BLOCK_SIZE * BLOCK_SIZE - 1) / (BLOCK_SIZE * BLOCK_SIZE);
    dim3 reductionGridDim(reduction_grid_size, 1);

    /* Flood simulation */
    for (*minute = 0; *minute < p->num_minutes; (*minute)++) {

        /* Step 1.1: Clouds movement (on CPU) */
        for (int cloud = 0; cloud < p->num_clouds; cloud++) {
            Cloud_t *c_cloud = &p->clouds[cloud];
            float km_minute = c_cloud->speed / 60.0f;
            c_cloud->x += km_minute * cosf(c_cloud->angle * M_PI / 180.0f);
            c_cloud->y += km_minute * sinf(c_cloud->angle * M_PI / 180.0f);
        }

#ifdef DEBUG
#ifndef ANIMATION
        print_clouds(p->num_clouds, p->clouds);
#endif
#endif

        /* Copy clouds to device */
        CUDA_CHECK_FUNCTION(cudaMemcpy(d_clouds, p->clouds, size_clouds, cudaMemcpyHostToDevice));

        /* Step 1.2: Rainfall */
        kernel_rainfall<<<gridDim, blockDim>>>(d_water_level, d_clouds, p->num_clouds, rows, columns,
                                                p->ex_factor, d_total_rain);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

#ifdef DEBUG
        CUDA_CHECK_FUNCTION(cudaMemcpy(h_water_level, d_water_level, size_grid, cudaMemcpyDeviceToHost));
        print_matrix(PRECISION_FIXED, rows, columns, h_water_level, "Water after rain");
#endif

        /* Step 2: Compute water spillage to neighbor cells */
        kernel_compute_spillage<<<gridDim, blockDim>>>(d_water_level, d_ground, d_spillage_flag, d_spillage_level,
                                                        d_spillage_from_neigh, rows, columns, d_total_water_loss);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

        /* Step 3: Propagation of previously computed water spillage to/from neighbors */
        // Reset max spillage iter for this minute
        double h_max_spillage_iter = 0.0;
        CUDA_CHECK_FUNCTION(cudaMemcpy(d_max_spillage_iter, &h_max_spillage_iter, sizeof(double), cudaMemcpyHostToDevice));

        // Propagate spillage and record per-cell maximums
        kernel_propagate_spillage<<<gridDim, blockDim>>>(d_water_level, d_spillage_flag, d_spillage_level,
                                                          d_spillage_from_neigh, rows, columns,
                                                          d_max_spillage_iter_cell);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

        // Reduce maximums and update global values
        kernel_reduce_max_spillage<<<reductionGridDim, blockDim>>>(d_max_spillage_iter_cell, rows * columns,
                                                                    d_max_spillage_iter, d_max_spillage_scenario,
                                                                    d_max_spillage_minute, *minute);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

        /* Copy max spillage back to check termination condition */
        CUDA_CHECK_FUNCTION(cudaMemcpy(&h_max_spillage_iter, d_max_spillage_iter, sizeof(double), cudaMemcpyDeviceToHost));

#ifdef DEBUG
#ifndef ANIMATION
        CUDA_CHECK_FUNCTION(cudaMemcpy(h_water_level, d_water_level, size_grid, cudaMemcpyDeviceToHost));
        print_matrix(PRECISION_FIXED, rows, columns, h_water_level, "Water after spillage");
#endif
#endif

        /* Reset ancillary structures */
        kernel_reset_structures<<<gridDim, blockDim>>>(d_spillage_flag, d_spillage_level, d_spillage_from_neigh, rows,
                                                        columns);
        CUDA_CHECK_KERNEL();
        CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

        /* Check termination condition */
        if (h_max_spillage_iter <= p->threshold) {
            (*minute)++;
            break;
        }
    }

    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

    r->runtime = get_time() - r->runtime;

    /* Copy results back to host */
    CUDA_CHECK_FUNCTION(cudaMemcpy(h_water_level, d_water_level, size_grid, cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&r->total_rain, d_total_rain, sizeof(long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&r->total_water_loss, d_total_water_loss, sizeof(long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&r->max_spillage_scenario, d_max_spillage_scenario, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&r->max_spillage_minute, d_max_spillage_minute, sizeof(int), cudaMemcpyDeviceToHost));

    if (p->final_matrix) {
        print_matrix(PRECISION_FIXED, rows, columns, h_water_level, "Water after spillage");
    }

    /* Statistics: Compute final statistics on GPU */
    kernel_compute_statistics<<<reductionGridDim, blockDim>>>(d_water_level, rows, columns, d_max_water_scenario, d_total_water);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

    CUDA_CHECK_FUNCTION(cudaMemcpy(&r->max_water_scenario, d_max_water_scenario, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_FUNCTION(cudaMemcpy(&r->total_water, d_total_water, sizeof(long), cudaMemcpyDeviceToHost));

    /* Free device memory */
    CUDA_CHECK_FUNCTION(cudaFree(d_water_level));
    CUDA_CHECK_FUNCTION(cudaFree(d_ground));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_flag));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_level));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_from_neigh));
    CUDA_CHECK_FUNCTION(cudaFree(d_max_spillage_iter_cell));
    CUDA_CHECK_FUNCTION(cudaFree(d_clouds));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_rain));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_water_loss));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_water));
    CUDA_CHECK_FUNCTION(cudaFree(d_max_spillage_iter));
    CUDA_CHECK_FUNCTION(cudaFree(d_max_spillage_scenario));
    CUDA_CHECK_FUNCTION(cudaFree(d_max_spillage_minute));
    CUDA_CHECK_FUNCTION(cudaFree(d_max_water_scenario));

    /* Free host memory */
    free(h_water_level);
    free(h_spillage_flag);
    free(h_spillage_level);
    free(h_spillage_from_neigh);

    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
}