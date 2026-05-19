#ifndef _KNN_GPU_H
#define _KNN_GPU_H


// exposes cuda implementations of knn
namespace knn_gpu{

__global__ void min_k_v2_fp32(const float* data, int start, int end, int row, float* result);

__global__ void min_k_fp32(const int N, const int* indptr, const int* indices,
                      const float* data, float* result);

__global__ void min_k_finprotax_fp32(const int N, const int* indptr, const int* indices,
                      const float* data, float* result);

__global__ void min_k_mean_fp32(const int N, const int* indptr, const int* indices,
                      const float* data, float* result);

__global__ void max_k_mean_fp32(const int N, const int* indptr, const int* indices,
                        const float* data, float* result);
  
__global__ void min_k_finprotax_warp_fp32(const int N, const int* indptr, const int* indices, const float* data, float* result);

// min + (min - q97) where q97 is an approximate 97% quantile; skips negative values
__global__ void min_q97_gap_warp_fp32(const int N, const int* indptr, const int* indices, const float* data, float* result);

// min + weighted (q - min) gap using a small-index quantile from sorted distances.
// Valid distances are v > 0.0f (strict), so masked negatives and zeros are skipped.
__global__ void min_q97_weighted_gap_warp_fp32(const int N, const int* indptr, const int* indices, const float* data, float* result);

}  // knn_gpu

#endif
