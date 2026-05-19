#include "knn_gpu.cuh"
#include <cfloat>

// parallel reduction version of min k
template<typename T>
__device__ void min_k_v2_impl(const float* data, int start, int end, int row, float* result){
    if (end-start <= 0){
        return;
    }
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ float sdata[];
    sdata[i] = data[i];

    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1){
        if (i < s){
            if (sdata[i] > sdata[i+s]){
                sdata[i] = sdata[i+s];
            }
        }
        __syncthreads();
    }

    if (i == 0){
        result[row*2] = sdata[0];
        result[row*2 + 1] = sdata[1];
    }
}

// --------------------------------------------------

// naive version of min k
template<typename T>
__device__ void min_k_impl(const int N, const int* indptr, const int* indices,
                 const T* data, T* result) {
    // get the row index, i.e. node index
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N){
        return;
    }
    
    // start-end segment for row
    int start = indptr[i];
    int end = indptr[i + 1];

    T m1 = 1;
    T m2 = 1;

    if (end-start <= 0){
        m2 = 0;
        m1 = 0;
    }
    else{
        m2 = data[start];
        m1 = data[start];
        start++;
    }

    // nonzero elements in row
    for (int j = start; j < end; j++) {
        T val = data[j];
        if (val < m1) {
            m2 = m1;
            m1 = val;
        }
        else if (val < m2){
          m2 = val;
        }
    }

    // store the minimum value in the result array
    result[i*2] = m1;
    result[i*2 + 1] = m2;
}

// --------------------------------------------------

// min k variant used by finprotax
// stores top 1, and diff with 2nd
template<typename T>
__device__ void min_k_finprotax_impl(const int N, const int* indptr, const int* indices,
                 const T* data, T* result) {
    // get the row index, i.e. node index
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N){
        return;
    }
    
    // start-end segment for row
    int start = indptr[i];
    int end = indptr[i + 1];

    T m1 = 1;
    T m2 = 1;

    if (end-start <= 0){
        m2 = 0;
        m1 = 0;
    }
    else{
        m2 = data[start];
        m1 = data[start];
        start++;
    }

    // nonzero values in this row
    for (int j = start; j < end; j++) {
        T val = data[j];
        if (val < m1) {
            m2 = m1;
            m1 = val;
        }
        else if (val < m2){
          m2 = val;
        }
    }
    m2 = m2-m1;

    // store the minimum value in the result array
    result[i*2] = m1;
    result[i*2 + 1] = m2;
}

// --------------------------------------------------
// min k faster variant
// --------------------------------------------------

__device__ __forceinline__ void combine_min2(float a1, float a2, float b1, float b2, float &o1, float &o2) {
    // Combine two (min1, min2) pairs into a new pair (o1, o2)
    float c1 = a1, c2 = a2;
    if (b1 < c1) {
        c2 = c1;
        c1 = b1;
    } else if (b1 < c2) {
        c2 = b1;
    }
    if (b2 < c1) {
        c2 = c1;
        c1 = b2;
    } else if (b2 < c2) {
        c2 = b2;
    }
    o1 = c1;
    o2 = c2;
}

__device__ void min_k_finprotax_warp(const int N,
                                          const int* __restrict__ indptr,
                                          const int* __restrict__ indices,  // unused
                                          const float* __restrict__ data,
                                          float* __restrict__ result) {
    const int lane      = threadIdx.x & 31;                 // 0..31
    const int warp_in_b = threadIdx.x >> 5;                 // warp id within block
    const int warps_per_block = blockDim.x >> 5;
    const int warp_global = blockIdx.x * warps_per_block + warp_in_b;

    if (warp_global >= N) return;
    const int i = warp_global;

    int start = indptr[i];
    int end   = indptr[i + 1];
    int len   = end - start;

    if (len <= 0) {
        if (lane == 0) {
            result[i*2]     = 0.0f;
            result[i*2 + 1] = 0.0f;
        }
        return;
    }

    // Each lane scans j = start + lane, j += warpSize (skip negative entries)
    float m1 = FLT_MAX;
    float m2 = FLT_MAX;
    int valid_count = 0;

    for (int j = start + lane; j < end; j += 32) {
        float v = data[j];
        if (v < 0.0f) {
            continue;
        }
        valid_count++;
        if (v < m1) {
            m2 = m1;
            m1 = v;
        } else if (v < m2) {
            m2 = v;
        }
    }

    // Warp reduction: combine local (m1,m2) across lanes.
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m1_other = __shfl_down_sync(mask, m1, offset);
        float m2_other = __shfl_down_sync(mask, m2, offset);
        float o1, o2;
        combine_min2(m1, m2, m1_other, m2_other, o1, o2);
        m1 = o1;
        m2 = o2;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        valid_count += __shfl_down_sync(mask, valid_count, offset);
    }

    if (lane == 0) {
        if (valid_count == 0) {
            result[i*2]     = 0.0f;
            result[i*2 + 1] = 0.0f;
        } else if (valid_count == 1) {
            result[i*2]     = m1;
            result[i*2 + 1] = 0.0f;
        } else {
            result[i*2]     = m1;
            result[i*2 + 1] = m2 - m1;
        }
    }
}

// --------------------------------------------------
// min + (min - q97) variant (skip negative values)
// q97 is approximated with a small per-warp histogram.
// --------------------------------------------------
__device__ void min_q97_gap_warp(const int N,
                                     const int* __restrict__ indptr,
                                     const int* __restrict__ indices,  // unused
                                     const float* __restrict__ data,
                                     float* __restrict__ result) {
    constexpr int BINS = 128;
    constexpr int MAX_WARPS_PER_BLOCK = 32;  // safe for THREADS_PER_BLOCK <= 1024

    const int lane      = threadIdx.x & 31;
    const int warp_in_b = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int warp_global = blockIdx.x * warps_per_block + warp_in_b;

    if (warp_global >= N) return;
    const int i = warp_global;

    const int start = indptr[i];
    const int end   = indptr[i + 1];
    const int len   = end - start;

    if (len <= 0) {
        if (lane == 0) {
            result[i*2]     = 0.0f;
            result[i*2 + 1] = 0.0f;
        }
        return;
    }

    float mn = FLT_MAX;
    float mx = -FLT_MAX;
    int valid_count = 0;

    for (int j = start + lane; j < end; j += 32) {
        float v = data[j];
        if (v < 0.0f) continue;
        valid_count++;
        mn = (v < mn) ? v : mn;
        mx = (v > mx) ? v : mx;
    }

    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float mn_other = __shfl_down_sync(mask, mn, offset);
        float mx_other = __shfl_down_sync(mask, mx, offset);
        mn = (mn_other < mn) ? mn_other : mn;
        mx = (mx_other > mx) ? mx_other : mx;
        valid_count += __shfl_down_sync(mask, valid_count, offset);
    }

    if (valid_count == 0) {
        if (lane == 0) {
            result[i*2]     = 0.0f;
            result[i*2 + 1] = 0.0f;
        }
        return;
    }

    __shared__ int hist[MAX_WARPS_PER_BLOCK][BINS];

    // Clear bins for this warp.
    for (int b = lane; b < BINS; b += 32) {
        hist[warp_in_b][b] = 0;
    }
    __syncwarp(mask);

    float q97 = mn;
    const float range = mx - mn;
    if (range > 0.0f) {
        const float inv = static_cast<float>(BINS) / range;

        // Fill histogram (2nd pass).
        for (int j = start + lane; j < end; j += 32) {
            float v = data[j];
            if (v < 0.0f) continue;
            int b = static_cast<int>((v - mn) * inv);
            if (b < 0) b = 0;
            if (b >= BINS) b = BINS - 1;
            atomicAdd(&hist[warp_in_b][b], 1);
        }
        __syncwarp(mask);

        if (lane == 0) {
            // Ceil(0.97 * valid_count) as a 1-based rank target.
            const int target = (97 * valid_count + 99) / 100;
            int cum = 0;
            int b97 = BINS - 1;
            for (int b = 0; b < BINS; ++b) {
                cum += hist[warp_in_b][b];
                if (cum >= target) {
                    b97 = b;
                    break;
                }
            }
            const float bin_w = range / static_cast<float>(BINS);
            q97 = mn + (static_cast<float>(b97) + 0.5f) * bin_w;
        }
        q97 = __shfl_sync(mask, q97, 0);
    }

    if (lane == 0) {
        result[i*2]     = mn;
        result[i*2 + 1] = mn - q97;
    }
}

// --------------------------------------------------
// min + weighted gap variant (skip v <= 0)
//
// Pseudocode:
//   min_distance = mn
//   n = valid_count
//   quantile_idx = 1 + int(n*0.03 + 0.5)
//   if quantile_idx > 10: quantile_idx = 10
//   if quantile_idx >= n: q = max
//   else: q = sorted_distances[quantile_idx]
//   weight = 1 - n^(-0.2)
//   gap = (q - mn) * weight
// --------------------------------------------------
__device__ __forceinline__ void insert_top11(float v, float (&a)[11]) {
    if (v >= a[10]) return;
    a[10] = v;
    #pragma unroll
    for (int k = 10; k > 0; --k) {
        if (a[k] < a[k - 1]) {
            float tmp = a[k - 1];
            a[k - 1] = a[k];
            a[k] = tmp;
        } else {
            break;
        }
    }
}

__device__ __forceinline__ void merge_top11(const float (&b)[11], float (&a)[11]) {
    float out[11];
    int ia = 0, ib = 0;
    #pragma unroll
    for (int k = 0; k < 11; ++k) {
        float va = a[ia];
        float vb = b[ib];
        if (vb < va) {
            out[k] = vb;
            ib = (ib < 10) ? (ib + 1) : 10;
        } else {
            out[k] = va;
            ia = (ia < 10) ? (ia + 1) : 10;
        }
    }
    #pragma unroll
    for (int k = 0; k < 11; ++k) a[k] = out[k];
}

__device__ void min_q97_weighted_gap_warp(const int N,
                                          const int* __restrict__ indptr,
                                          const int* __restrict__ indices,  // unused
                                          const float* __restrict__ data,
                                          float* __restrict__ result) {
    const int lane      = threadIdx.x & 31;
    const int warp_in_b = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int warp_global = blockIdx.x * warps_per_block + warp_in_b;

    if (warp_global >= N) return;
    const int i = warp_global;

    const int start = indptr[i];
    const int end   = indptr[i + 1];
    const int len   = end - start;

    if (len <= 0) {
        if (lane == 0) {
            result[i*2]     = 0.0f;
            result[i*2 + 1] = 0.0f;
        }
        return;
    }

    float top[11];
    #pragma unroll
    for (int k = 0; k < 11; ++k) top[k] = FLT_MAX;

    float mx = -FLT_MAX;
    int valid_count = 0;

    for (int j = start + lane; j < end; j += 32) {
        float v = data[j];
        // strict > 0: skip masked negatives and also avoid -0.0 and 0.0
        if (!(v > 0.0f)) continue;
        valid_count++;
        mx = (v > mx) ? v : mx;
        insert_top11(v, top);
    }

    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float mx_other = __shfl_down_sync(mask, mx, offset);
        mx = (mx_other > mx) ? mx_other : mx;

        int c_other = __shfl_down_sync(mask, valid_count, offset);
        valid_count += c_other;

        float other[11];
        #pragma unroll
        for (int k = 0; k < 11; ++k) {
            other[k] = __shfl_down_sync(mask, top[k], offset);
        }
        merge_top11(other, top);
    }

    if (lane == 0) {
        if (valid_count <= 0) {
            result[i*2]     = 0.0f;
            result[i*2 + 1] = 0.0f;
            return;
        }

        const float mn = top[0];
        int quantile_idx = 1 + static_cast<int>(static_cast<float>(valid_count) * 0.03f + 0.5f);
        if (quantile_idx > 10) quantile_idx = 10;

        float q;
        if (quantile_idx >= valid_count) {
            q = mx;
        } else {
            // top[] contains the 11 smallest values; quantile_idx is capped at 10, so it's always present.
            q = top[quantile_idx];
        }

        const float n = static_cast<float>(valid_count);
        const float weight = 1.0f - powf(n, -0.2f);
        const float gap = (q - mn) * weight;

        result[i*2]     = mn;
        result[i*2 + 1] = gap;
    }
}

// --------------------------------------------------
// min k mean variant
// --------------------------------------------------
template<typename T>
// Gets the minimum and mean for each row.
__device__ void min_k_mean_impl(const int N, const int* indptr, const int* indices,
                 const T* data, T* result) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N){
        return;
    }

    int start = indptr[i];
    int end = indptr[i + 1];
    int count = end - start;

    T m1 = 1;
    T sum = 0;

    if (count <= 0){
        result[i*2] = 0;
        result[i*2 + 1] = 0;
        return;
    }

    m1 = data[start];
    sum = data[start];
    start++;

    for (int j = start; j < end; j++) {
        T val = data[j];
        sum += val;
        if (val < m1) {
            m1 = val;
        }
    }

    result[i*2] = m1;
    result[i*2 + 1] = sum / static_cast<T>(count);
}

template<typename T>
// Gets the max and mean for each row.
__device__ void max_k_mean_impl(const int N, const int* indptr, const int* indices,
                 const T* data, T* result) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N){
        return;
    }

    int start = indptr[i];
    int end = indptr[i + 1];
    int count = end - start;

    if (count <= 0){
        result[i*2] = 0;
        result[i*2 + 1] = 0;
        return;
    }

    T m1 = data[start];
    T sum = data[start];
    start++;

    for (int j = start; j < end; j++) {
        T val = data[j];
        sum += val;
        if (val > m1) {
            m1 = val;
        }
    }

    result[i*2] = m1;
    result[i*2 + 1] = sum / static_cast<T>(count);
}

// --------------------------------------------------

// define kernels specified in knn_gpu.cuh
namespace knn_gpu{

__global__ void min_k_finprotax_fp32(const int N, const int* indptr, const int* indices,
                      const float* data, float* result){
    // this will be inlined by nvcc
    min_k_finprotax_impl<float>(N, indptr, indices, data, result);
}

__global__ void min_k_finprotax_warp_fp32(const int N, const int* indptr, const int* indices,
                      const float* data, float* result){
    min_k_finprotax_warp(N, indptr, indices, data, result);
}

__global__ void min_q97_gap_warp_fp32(const int N, const int* indptr, const int* indices,
                      const float* data, float* result){
    min_q97_gap_warp(N, indptr, indices, data, result);
}

__global__ void min_q97_weighted_gap_warp_fp32(const int N, const int* indptr, const int* indices,
                      const float* data, float* result){
    min_q97_weighted_gap_warp(N, indptr, indices, data, result);
}

__global__ void min_k_mean_fp32(const int N, const int* indptr, const int* indices,
                      const float* data, float* result){
    min_k_mean_impl<float>(N, indptr, indices, data, result);
}

__global__ void max_k_mean_fp32(const int N, const int* indptr, const int* indices,
                      const float* data, float* result){
    max_k_mean_impl<float>(N, indptr, indices, data, result);
}

// -------------------------------------------------- Above are the used functions

__global__ void min_k_fp32(const int N, const int* indptr, const int* indices,
                      const float* data, float* result){
    min_k_impl<float>(N, indptr, indices, data, result);
}


__global__ void min_k_v2_fp32(const float* data, int start, int end, int row, float* result){
    min_k_v2_impl<float>(data, start, end, row, result);
}


}  // namespace knn_gpu

