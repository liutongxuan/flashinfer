/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <thrust/device_vector.h>

#include "flashinfer_ops.cuh"

using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;

void perf_flashinfer_single_decode(bool opt) {
  size_t seq_len = 128;
  size_t num_qo_heads = 32;
  size_t num_kv_heads = 32;
  size_t head_dim = 128;
  size_t pos_encoding_mode = 0;
  size_t kv_layout = 0;
  bool cooperative = true;
  // Allocate input data:
  thrust::device_vector<half> Q(num_qo_heads * head_dim);
  thrust::device_vector<half> K(seq_len * num_kv_heads * head_dim);
  thrust::device_vector<half> V(seq_len * num_kv_heads * head_dim);
  thrust::device_vector<half> O(num_qo_heads * head_dim);
  thrust::device_vector<half> tmp(16 * 1024 * 1024);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  // Provide throughput information:
  cudaError_t status = flashinfer::SingleDecodeWithKVCache(
      thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()),
      thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()),
      cooperative ? thrust::raw_pointer_cast(tmp.data()) : nullptr, num_qo_heads, num_kv_heads,
      seq_len, head_dim, QKVLayout(kv_layout), PosEncodingMode(pos_encoding_mode),
      /*maybe_sm_scale=*/std::nullopt,
      /*rope_scale=*/1.f,
      /*rope_theta=*/1e4, stream, opt);
  if (status != cudaSuccess) {
    std::cout << "Execution error" << std::endl;
  }
}

int main() {
  // baseline
  perf_flashinfer_single_decode(false);
  // optimized
  perf_flashinfer_single_decode(true);
  return 0;
}
