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
#include <iostream>
#include <thrust/device_vector.h>

#include "flashinfer_ops.cuh"

using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;

struct decode_input_data {
  size_t seq_len = 1024;
  size_t num_qo_heads = 32;
  size_t num_kv_heads = 32;
  size_t head_dim = 128;
  size_t pos_encoding_mode = 0;
  size_t kv_layout = 0;
  bool cooperative = true;

  thrust::device_vector<half>* Q = nullptr;
  thrust::device_vector<half>* K = nullptr;
  thrust::device_vector<half>* V = nullptr;
  thrust::device_vector<half>* O = nullptr;
  thrust::device_vector<half>* tmp = nullptr;
  // Allocate input data:
  decode_input_data() {
    Q = new thrust::device_vector<half>(num_qo_heads * head_dim);
    K = new thrust::device_vector<half>(seq_len * num_kv_heads * head_dim);
    V = new thrust::device_vector<half>(seq_len * num_kv_heads * head_dim);
    O = new thrust::device_vector<half>(num_qo_heads * head_dim);
    tmp = new thrust::device_vector<half>(16 * 1024 * 1024);
  }

  ~decode_input_data() {
    delete tmp;
    delete O;
    delete V;
    delete K;
    delete Q;
  }
};

struct prefill_input_data {
  size_t kv_len = 2048;
  size_t qo_len = kv_len;
  size_t num_kv_heads = 32;
  size_t num_qo_heads = 32;
  size_t head_dim = 128;
  size_t pos_encoding_mode = 0;
  size_t kv_layout = 0;
  bool causal = false;
  bool cooperative = true;
  bool allow_fp16_qk_reduction = false;

  // Allocate input data:
  thrust::device_vector<half>* Q = nullptr;
  thrust::device_vector<half>* K = nullptr;
  thrust::device_vector<half>* V = nullptr;
  thrust::device_vector<uint8_t>* mask = nullptr;
  thrust::device_vector<half>* O = nullptr;
  thrust::device_vector<half>* tmp = nullptr;
  
  prefill_input_data() {
    Q = new thrust::device_vector<half>(qo_len * num_qo_heads * head_dim);
    K = new thrust::device_vector<half>(kv_len * num_kv_heads * head_dim);
    V = new thrust::device_vector<half>(kv_len * num_kv_heads * head_dim);
    mask = new thrust::device_vector<uint8_t>(qo_len * kv_len / 8);
    O = new thrust::device_vector<half>(qo_len * num_qo_heads * head_dim);
    tmp = new thrust::device_vector<half>(16 * 1024 * 1024);
  }

  ~prefill_input_data() {
    delete tmp;
    delete O;
    delete mask;
    delete V;
    delete K;
    delete Q;
  }
};

void perf_flashinfer_single_decode(cudaStream_t& stream, decode_input_data* input, bool opt) {
  // Provide throughput information:
  cudaError_t status = flashinfer::SingleDecodeWithKVCache(
      thrust::raw_pointer_cast(input->Q->data()), thrust::raw_pointer_cast(input->K->data()),
      thrust::raw_pointer_cast(input->V->data()), thrust::raw_pointer_cast(input->O->data()),
      input->cooperative ? thrust::raw_pointer_cast(input->tmp->data()) : nullptr, input->num_qo_heads, input->num_kv_heads,
      input->seq_len, input->head_dim, QKVLayout(input->kv_layout), PosEncodingMode(input->pos_encoding_mode),
      /*maybe_sm_scale=*/std::nullopt,
      /*rope_scale=*/1.f,
      /*rope_theta=*/1e4, stream, opt);
  if (status != cudaSuccess) {
    std::cout << "Execution error" << std::endl;
  }
}

void perf_flashinfer_single_prefill(cudaStream_t& stream, prefill_input_data* input, bool opt) {
  auto status = flashinfer::SinglePrefillWithKVCache<half, half>(
      thrust::raw_pointer_cast(input->Q->data()), thrust::raw_pointer_cast(input->K->data()),
      thrust::raw_pointer_cast(input->V->data()), thrust::raw_pointer_cast(input->O->data()),
      input->cooperative ? thrust::raw_pointer_cast(input->tmp->data()) : nullptr,
      nullptr, input->num_qo_heads, input->num_kv_heads, input->qo_len, input->kv_len, input->head_dim,
      input->causal, QKVLayout(input->kv_layout), PosEncodingMode(input->pos_encoding_mode),
      input->allow_fp16_qk_reduction, std::nullopt, 1.f, 1e4, stream, opt);

  if (status != cudaSuccess) {
    std::cout << "Execution error" << std::endl;
  }
}

int main() {
  decode_input_data decode_data;
  cudaStream_t decode_stream;
  cudaStreamCreate(&decode_stream);
  prefill_input_data prefill_data;
  cudaStream_t prefill_stream;
  cudaStreamCreate(&prefill_stream);

  for (int i = 0; i < 100; ++i) {
    perf_flashinfer_single_decode(decode_stream, &decode_data, false);
    perf_flashinfer_single_prefill(prefill_stream, &prefill_data, false);
  }
  cudaStreamSynchronize(decode_stream);
  cudaStreamSynchronize(prefill_stream);
  return 0;
}
