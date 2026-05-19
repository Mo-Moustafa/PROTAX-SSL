// creates bindings for gpu ops
#include "gpu_ops.h"
#include "pybind11_kernel_helpers.h"

namespace {

// python dict holding function pointers
pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["gpu_knn_f32"] = xla_helpers::EncapsulateFunction(knn::gpu_knn_f32);
  dict["gpu_knn_v2_f32"] = xla_helpers::EncapsulateFunction(knn::gpu_knn_v2_f32);
  dict["gpu_knn_mean_f32"] = xla_helpers::EncapsulateFunction(knn::gpu_knn_mean_f32);                      // This is the min and mean version
  dict["gpu_knn_max_mean_f32"] = xla_helpers::EncapsulateFunction(knn::gpu_knn_max_mean_f32);               // This is the max and mean version
  dict["gpu_knn_finprotax_warp_f32"] = xla_helpers::EncapsulateFunction(knn::gpu_knn_finprotax_warp_f32);   // This is the enhanced min and gap version
  dict["gpu_knn_q97_gap_warp_f32"] = xla_helpers::EncapsulateFunction(knn::gpu_knn_q97_gap_warp_f32);       // min and (min - q97), skipping negatives
  dict["gpu_knn_q97_weighted_gap_warp_f32"] = xla_helpers::EncapsulateFunction(knn::gpu_knn_q97_weighted_gap_warp_f32);  // weighted gap using small-index quantile
  return dict;
}

// define python module: gpu ops
// expose registrations dict and build_knn_descriptor()
PYBIND11_MODULE(gpu_ops, m){
    m.def("registrations", &Registrations);
    m.def("build_knn_descriptor",
          [](int rows) { return xla_helpers::PackDescriptor(knn::KNNDescriptor{rows, 2, 1}); });
    m.def("build_knn_descriptor_batched",
          [](int rows, int batch_size) { return xla_helpers::PackDescriptor(knn::KNNDescriptor{rows, 2, batch_size}); });
}
} 

