#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>

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

#include "rng.c"
#include "flood.h"

extern "C" double get_time();


typedef struct {
    float x0, y0;
    float dx, dy;
    float radius;
    float intensity;
} CloudPrecomp_t;

/*
 * Integer bounding box for a cloud at a given minute, in matrix coordinates.
 * Precomputed on host each minute so the rainfall kernel can skip cells outside
 * all clouds' bounding boxes cheaply with integer comparisons.
 */
typedef struct {
    int row_min, row_max; /* inclusive range of rows covered */
    int col_min, col_max; /* inclusive range of cols covered */
} CloudBBox_t;


/* float atomicMax */
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

/* Kernel 1: Rainfall  
 * Computes rain contribution from a single cloud to cell (row, col).
 * Bounding box passed as integers so we can skip non-covered cells with
 * a cheap integer comparison before doing any float math.
 */
__device__ float rainFromCloud(int minute, int row, int col,
                               const CloudPrecomp_t *cloud,
                               const CloudBBox_t *bbox,
                               float ex_factor,
                               int rows, int columns) {
    /* Integer bounding-box rejection: skip cells clearly outside the cloud */
    if (row < bbox->row_min || row > bbox->row_max ||
        col < bbox->col_min || col > bbox->col_max)
        return 0.0f;

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
                                const CloudPrecomp_t *clouds,
                                const CloudBBox_t *bboxes,
                                int num_clouds,
                                float ex_factor, int rows, int columns,
                                long *d_total_rain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * columns) return;

    int row = idx / columns;
    int col = idx % columns;

    int total = 0;
    for (int c = 0; c < num_clouds; c++) {
        float rain = rainFromCloud(minute, row, col, &clouds[c], &bboxes[c],
                                   ex_factor, rows, columns);
        if (rain > 0.0f)
            total += FIXED(rain);
    }
    if (total > 0) {
        water_level[idx] += total;
        atomicAdd((unsigned long long *)d_total_rain, (unsigned long long)total);
    }
}

/*  Kernel 2: Spillage computation  
 * Each thread computes spillage for its own cell and writes to 5 SoA planes.
 * All writes are isolated to the thread's own cell index, no conflicts.
 * Boundary spillage goes to d_total_water_loss via atomicAdd.
 * The kernel also serves as the "reset" for the spillage planes by
 * unconditionally zeroing all 5 slots before any early return.
 */
__global__ void spillage_compute_kernel(const int *water_level, const float *ground,
                                        int *spill_self,
                                        int *spill_top, int *spill_bottom,
                                        int *spill_left, int *spill_right,
                                        int rows, int columns,
                                        long *d_total_water_loss) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * columns) return;

    int row = idx / columns;
    int col = idx % columns;

    /* Reset all planes for this cell (doubles as the "memset" each iteration) */
    spill_self[idx] = 0;
    spill_top[idx] = 0;
    spill_bottom[idx] = 0;
    spill_left[idx] = 0;
    spill_right[idx] = 0;

    if (water_level[idx] <= 0) return;

    float current_height = ground[idx] + FLOATING(water_level[idx]);
    float sum_diff = 0.0f;
    float my_spillage_level = 0.0f;

    /* Height differences for all 4 directions */
    float height_diffs[4];

    for (int d = 0; d < CONTIGUOUS_CELLS; d++) {
        int nr = row + displacements[d][0];
        int nc = col + displacements[d][1];

        float neighbor_height;
        if (nr < 0 || nr >= rows || nc < 0 || nc >= columns)
            /* boundary: same ground, no water */    
            neighbor_height = ground[idx]; 
        else
            neighbor_height = ground[nr * columns + nc] + FLOATING(water_level[nr * columns + nc]);

        if (current_height >= neighbor_height) {
            float hd = current_height - neighbor_height;
            height_diffs[d] = hd;
            sum_diff += hd;
            my_spillage_level = fmaxf(my_spillage_level, hd);
        } else {
            height_diffs[d] = 0.0f;
        }
    }

    my_spillage_level = fminf(FLOATING(water_level[idx]), my_spillage_level);

    if (sum_diff <= 0.0f) return;

    float proportion = my_spillage_level / sum_diff;
    if (proportion <= 1e-8f) return;

    /* Self loss (negative, fixed-point, already divided by SPILLAGE_FACTOR) */
    spill_self[idx] = -FIXED(my_spillage_level / SPILLAGE_FACTOR);

    /* Per-direction output arrays, indexed by direction */
    int *spill_planes[4] = {spill_top, spill_bottom, spill_left, spill_right};

    for (int d = 0; d < CONTIGUOUS_CELLS; d++) {
        if (height_diffs[d] <= 0.0f) continue;

        int nr = row + displacements[d][0];
        int nc = col + displacements[d][1];

        if (nr < 0 || nr >= rows || nc < 0 || nc >= columns) {
            /* Boundary: water leaves the simulation */
            long loss = FIXED(proportion * height_diffs[d] / SPILLAGE_FACTOR);
            atomicAdd((unsigned long long *)d_total_water_loss, (unsigned long long)loss);
        } else {
            /* In-bounds: store contribution in this cell's slot of the SoA plane */
            spill_planes[d][idx] = FIXED(proportion * height_diffs[d] / SPILLAGE_FACTOR);
        }
    }
}

/*  Kernel 3: Propagation
 * Each thread gathers the net water change for its cell:
 *   1. Self loss from spill_self[idx]
 *   2. What top neighbor spills toward me:    spill_bottom[idx - columns]
 *   3. What bottom neighbor spills toward me: spill_top[idx + columns]
 *   4. What left neighbor spills toward me:   spill_right[idx - 1]
 *   5. What right neighbor spills toward me:  spill_left[idx + 1]
 *
 * Because each plane is contiguous, adjacent threads read adjacent addresses
 * = fully coalesced global memory reads.
 *
 * Block-level shared-memory reduction computes the per-block max spillage,
 * then one atomicMaxFloat per block writes to d_max_spillage.
 */
__global__ void propagation_kernel(int *water_level,
                                   const int *spill_self,
                                   const int *spill_top, const int *spill_bottom,
                                   const int *spill_left, const int *spill_right,
                                   int rows, int columns,
                                   float *d_max_spillage) {
    extern __shared__ float sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total_cells = rows * columns;

    float my_spillage = 0.0f;

    if (idx < total_cells) {
        int row = idx / columns;
        int col = idx % columns;

        int self_loss = spill_self[idx];
        if (self_loss < 0)
            my_spillage = FLOATING(-self_loss);

        int net_change = self_loss;

        /* Gather from top neighbor (row-1) */
        if (row > 0)
            net_change += spill_bottom[idx - columns];

        /* Gather from bottom neighbor (row+1) */
        if (row < rows - 1)
            net_change += spill_top[idx + columns];

        /* Gather from left neighbor (col-1) */
        if (col > 0)
            net_change += spill_right[idx - 1];

        /* Gather from right neighbor (col+1) */
        if (col < columns - 1)
            net_change += spill_left[idx + 1];

        water_level[idx] += net_change;
    }

    /* Block-level reduction for max_spillage (avoids per-thread atomics) */
    sdata[tid] = my_spillage;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0 && sdata[0] > 0.0f)
        atomicMaxFloat(d_max_spillage, sdata[0]);
}

/* === Main host function === */

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

    /* Host-side bounding box buffer, recomputed each minute on the CPU */
    CloudBBox_t *bboxes = (CloudBBox_t *)malloc(sizeof(CloudBBox_t) * p->num_clouds);

    /* Device memory: grid arrays */
    int *d_water_level;
    float *d_ground;

    CUDA_CHECK_FUNCTION(cudaMalloc(&d_water_level, sizeof(int) * grid_size));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_ground, sizeof(float) * grid_size));
    CUDA_CHECK_FUNCTION(cudaMemset(d_water_level, 0, sizeof(int) * grid_size));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_ground, p->ground, sizeof(float) * grid_size, cudaMemcpyHostToDevice));

    /* SoA spillage planes: 5 separate arrays, each grid_size ints */
    int *d_spill_self, *d_spill_top, *d_spill_bottom, *d_spill_left, *d_spill_right;
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_spill_self,   sizeof(int) * grid_size));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_spill_top,    sizeof(int) * grid_size));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_spill_bottom, sizeof(int) * grid_size));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_spill_left,   sizeof(int) * grid_size));
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_spill_right,  sizeof(int) * grid_size));

    /* Device memory: clouds and bounding boxes */
    CloudPrecomp_t *d_clouds_pre;
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_clouds_pre, sizeof(CloudPrecomp_t) * p->num_clouds));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_clouds_pre, clouds_pre,
                                   sizeof(CloudPrecomp_t) * p->num_clouds, cudaMemcpyHostToDevice));

    CloudBBox_t *d_bboxes;
    CUDA_CHECK_FUNCTION(cudaMalloc(&d_bboxes, sizeof(CloudBBox_t) * p->num_clouds));

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


    float *h_max_spillage;
    CUDA_CHECK_FUNCTION(cudaMallocHost(&h_max_spillage, sizeof(float)));
    *h_max_spillage = 0.0f;

    /* Separate stream for the max_spillage D2H transfer so it can overlap
     * with CPU work (bbox computation) and the next iteration's H2D uploads. */
    cudaStream_t copy_stream;
    CUDA_CHECK_FUNCTION(cudaStreamCreate(&copy_stream));


    cudaEvent_t propagation_done;
    CUDA_CHECK_FUNCTION(cudaEventCreate(&propagation_done));

    int threads_per_block = 256;
    int num_blocks = ((int)grid_size + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = threads_per_block * sizeof(float);

#ifdef DEBUG
    print_matrix(PRECISION_FLOAT, rows, columns, p->ground, "Ground heights");
#ifndef ANIMATION
    print_clouds(p->num_clouds, p->clouds);
#endif
#endif

    double max_spillage_iter = DBL_MAX;
    int copy_pending = 0; /* whether an async D2H copy is in flight */

    r->runtime = get_time();

    /* Main loop.
     */
    for (*minute = 0; *minute < p->num_minutes && max_spillage_iter > p->threshold; (*minute)++) {

        /* Precompute per-cloud integer bounding boxes on host
         * Done FIRST so it overlaps with the in-flight async D2H copy
         * of max_spillage from the previous iteration. */
        for (int c = 0; c < p->num_clouds; c++) {
            float cx = clouds_pre[c].x0 + (*minute + 1) * clouds_pre[c].dx;
            float cy = clouds_pre[c].y0 + (*minute + 1) * clouds_pre[c].dy;
            float rad = clouds_pre[c].radius;

            float fmin_col = (float)columns * MAX(0.0f, cx - rad) / SCENARIO_SIZE;
            float fmax_col = (float)columns * MIN((float)SCENARIO_SIZE, cx + rad) / SCENARIO_SIZE;
            float fmin_row = (float)rows    * MAX(0.0f, cy - rad) / SCENARIO_SIZE;
            float fmax_row = (float)rows    * MIN((float)SCENARIO_SIZE, cy + rad) / SCENARIO_SIZE;

            bboxes[c].col_min = MAX(0, (int)floorf(fmin_col));
            bboxes[c].col_max = MIN(columns - 1, (int)floorf(fmax_col));
            bboxes[c].row_min = MAX(0, (int)floorf(fmin_row));
            bboxes[c].row_max = MIN(rows - 1, (int)floorf(fmax_row));
        }

        /* Consume previous iteration's async max_spillage result  */
        if (copy_pending) {
            CUDA_CHECK_FUNCTION(cudaStreamSynchronize(copy_stream));
            max_spillage_iter = (double)*h_max_spillage;

            if (*h_max_spillage > r->max_spillage_scenario) {
                r->max_spillage_scenario = *h_max_spillage;
                r->max_spillage_minute = *minute - 1;
            }

            /* Check if the previous iteration converged */
            if (max_spillage_iter <= p->threshold)
                break;

            copy_pending = 0;
        }

        /* Upload bboxes to device */
        CUDA_CHECK_FUNCTION(cudaMemcpyAsync(d_bboxes, bboxes,
                                            sizeof(CloudBBox_t) * p->num_clouds,
                                            cudaMemcpyHostToDevice, 0));

        /* Kernel A: Rainfall */
        rainfall_kernel<<<num_blocks, threads_per_block>>>(
            *minute, d_water_level, d_clouds_pre, d_bboxes, p->num_clouds,
            p->ex_factor, rows, columns, d_total_rain);
        CUDA_CHECK_KERNEL();

#ifdef DEBUG
        CUDA_CHECK_FUNCTION(cudaMemcpy(water_level, d_water_level,
                                       sizeof(int) * grid_size, cudaMemcpyDeviceToHost));
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after rain");
#endif

        /* Kernel B: Spillage computation (SoA planes, isolated writes per cell) */
        spillage_compute_kernel<<<num_blocks, threads_per_block>>>(
            d_water_level, d_ground,
            d_spill_self, d_spill_top, d_spill_bottom, d_spill_left, d_spill_right,
            rows, columns, d_total_water_loss);
        CUDA_CHECK_KERNEL();

        /* Zero d_max_spillage asynchronously before propagation */
        CUDA_CHECK_FUNCTION(cudaMemsetAsync(d_max_spillage, 0, sizeof(float), 0));

        /* Kernel C: Propagation (SoA gather + block-level max reduction) */
        propagation_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
            d_water_level,
            d_spill_self, d_spill_top, d_spill_bottom, d_spill_left, d_spill_right,
            rows, columns, d_max_spillage);
        CUDA_CHECK_KERNEL();

        /* Async readback of max_spillage */
        CUDA_CHECK_FUNCTION(cudaEventRecord(propagation_done, 0));
        CUDA_CHECK_FUNCTION(cudaStreamWaitEvent(copy_stream, propagation_done, 0));
        CUDA_CHECK_FUNCTION(cudaMemcpyAsync(h_max_spillage, d_max_spillage,
                                            sizeof(float), cudaMemcpyDeviceToHost,
                                            copy_stream));
        copy_pending = 1;

#ifdef DEBUG
#ifndef ANIMATION
        CUDA_CHECK_FUNCTION(cudaMemcpy(water_level, d_water_level,
                                       sizeof(int) * grid_size, cudaMemcpyDeviceToHost));
        print_matrix(PRECISION_FIXED, rows, columns, water_level, "Water after spillage");
#endif
#endif
    }

    /* Drain any pending async copy from the last iteration
     * This handles the case where the loop exits due to the time limit
     * (num_minutes reached) rather than convergence. The last iteration's
     * result is still in flight on copy_stream. */
    if (copy_pending) {
        CUDA_CHECK_FUNCTION(cudaStreamSynchronize(copy_stream));

        if (*h_max_spillage > r->max_spillage_scenario) {
            r->max_spillage_scenario = *h_max_spillage;
            /* remove the increment from the loop */
            r->max_spillage_minute = *minute - 1;
        }
    }

    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
    r->runtime = get_time() - r->runtime;

    /* Copy final water_level back to host */
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
    CUDA_CHECK_FUNCTION(cudaFree(d_spill_self));
    CUDA_CHECK_FUNCTION(cudaFree(d_spill_top));
    CUDA_CHECK_FUNCTION(cudaFree(d_spill_bottom));
    CUDA_CHECK_FUNCTION(cudaFree(d_spill_left));
    CUDA_CHECK_FUNCTION(cudaFree(d_spill_right));
    CUDA_CHECK_FUNCTION(cudaFree(d_clouds_pre));
    CUDA_CHECK_FUNCTION(cudaFree(d_bboxes));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_rain));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_water_loss));
    CUDA_CHECK_FUNCTION(cudaFree(d_max_spillage));
    CUDA_CHECK_FUNCTION(cudaEventDestroy(propagation_done));
    CUDA_CHECK_FUNCTION(cudaStreamDestroy(copy_stream));

    /* Free host resources */
    free(clouds_pre);
    free(bboxes);
    free(water_level);
    CUDA_CHECK_FUNCTION(cudaFreeHost(h_max_spillage));
}
