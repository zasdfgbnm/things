#include <cub/cub.cuh>
#include <chrono>

template <int N> struct alignas(N) OpaqueType { char data[N]; };

using KeyT = float;
using ValueT = OpaqueType<8>;
constexpr size_t N = 100'000'000;

KeyT h_keys[N];
ValueT h_values[N];

void init() {
  for (size_t i = 0; i < N; i++) {
    h_keys[i] = (i % 10);
    for (auto &c : h_values[i].data) {
      c = (char)i;
    }
  }
}

int main() {
  KeyT *d_keys;
  cudaMalloc(&d_keys, sizeof(KeyT) * N);
  cudaMemcpy(d_keys, h_keys, sizeof(KeyT) * N, cudaMemcpyDefault);

  ValueT *d_values;
  cudaMalloc(&d_values, sizeof(ValueT) * N);
  cudaMemcpy(d_values, h_values, sizeof(ValueT) * N, cudaMemcpyDefault);

  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, d_keys, d_keys, d_values, d_values, N);

  void *d_temp_storage;
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  cudaDeviceSynchronize();
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys, d_values, d_values, N);
  cudaDeviceSynchronize();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
}

// nsys nvprof nvcc -std=c++14 -arch=sm_80 -run cub-sort.cu
// nvcc -std=c++14 -arch=sm_80 -run cub-sort.cu