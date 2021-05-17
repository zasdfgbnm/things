#include <vector>
#include <cassert>

constexpr int64_t DIMS = 4;

struct Tensor {
    int64_t shape[DIMS];
    int64_t strides[DIMS];
    float *data;
    float & operator()(int64_t i, int64_t j, int64_t k, int64_t l) {
        int64_t index = i * strides[0] + j * strides[1] + k * strides[2] + l * strides[3];
        return data[index];
    }
};

Tensor new_tensor(const std::vector<int64_t> &shape, const std::vector<int64_t> &dim_order) {
    Tensor ret;

    for(int i = 0; i < DIMS; i++) {
        ret.shape[i] = shape[i];
    }

    int64_t size = 1;
    for(int i = 0; i < DIMS; i++) {
        auto dim = dim_order[i];
        ret.strides[dim] = size;
        size *= shape[dim];
    }
    cudaMallocManaged(&ret.data, size * sizeof(float));
    return ret;
}

void copy(Tensor &to, Tensor &from) {
    for (int i = 0; i < from.shape[0]; i++) {
        for (int j = 0; j < from.shape[1]; j++) {
            for (int k = 0; k < from.shape[2]; k++) {
                for (int l = 0; l < from.shape[3]; l++) {
                    to(i, j, k, l) = from(i, j, k, l);
                }
            }
        }
    }
}

int main() {}