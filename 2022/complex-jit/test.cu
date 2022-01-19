#include "complex"

__global__ void kernel(std::complex<float> *in, std::complex<float> *out) {
    *out = std::sin(*in);
}

int main() {
    kernel<<<1,1>>>(nullptr, nullptr);
}
